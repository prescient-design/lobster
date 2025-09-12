import logging
from collections.abc import Sequence

import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from lobster.constants import (
    BIOPYTHON_FEATURE_AGGREGATION_METHODS,
    BIOPYTHON_FEATURES,
    BIOPYTHON_PEPTIDE_SCALER_PARAMS,
    BIOPYTHON_PROTEIN_SCALER_PARAMS,
    PEPTIDE_WARNING_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _split_protein_complex(sequence: str, separator: str) -> list[str]:
    """Split protein complex into individual chains."""
    chains = [chain.strip() for chain in sequence.split(separator)]

    for i, chain in enumerate(chains):
        if not chain:
            raise ValueError(f"Empty chain found at position {i} after splitting sequence with '{separator}'")

    return chains


def _aggregate_features(chain_features: list[dict[str, float]], feature_list: list[str]) -> dict[str, float]:
    """Aggregate features across chains using predefined aggregation methods."""
    if not chain_features:
        raise ValueError("No valid chains found in protein complex")

    aggregated = {}

    for feature in feature_list:
        values = [chain_feat[feature] for chain_feat in chain_features]
        aggregation_func = BIOPYTHON_FEATURE_AGGREGATION_METHODS[feature]
        aggregated[feature] = aggregation_func(values)

    return aggregated


def _compute_single_chain_features(sequence: str, feature_list: list[str]) -> dict[str, float]:
    """Compute BioPython features for a single protein chain."""
    feature_set = set(feature_list)
    protein_analysis = ProteinAnalysis(sequence)
    computed_features = {}

    # Sequence length
    if "sequence_length" in feature_set:
        computed_features["sequence_length"] = len(sequence)

    # Simple features (single method calls)
    simple_features = {
        "molecular_weight": "molecular_weight",
        "aromaticity_index": "aromaticity",
        "instability_index": "instability_index",
        "isoelectric_point": "isoelectric_point",
        "grand_average_hydropathy_index": "gravy",
    }

    for feature_name, method_name in simple_features.items():
        if feature_name in feature_set:
            computed_features[feature_name] = getattr(protein_analysis, method_name)()

    # pH charge features
    if "net_charge_at_ph_6" in feature_set:
        computed_features["net_charge_at_ph_6"] = protein_analysis.charge_at_pH(6)
    if "net_charge_at_ph_7" in feature_set:
        computed_features["net_charge_at_ph_7"] = protein_analysis.charge_at_pH(7)

    # Expensive operations - compute once if any related feature is needed
    secondary_structure_features = {"alpha_helix_fraction", "turn_structure_fraction", "beta_sheet_fraction"}
    if feature_set & secondary_structure_features:
        fractions = protein_analysis.secondary_structure_fraction()
        if "alpha_helix_fraction" in feature_set:
            computed_features["alpha_helix_fraction"] = fractions[0]
        if "turn_structure_fraction" in feature_set:
            computed_features["turn_structure_fraction"] = fractions[1]
        if "beta_sheet_fraction" in feature_set:
            computed_features["beta_sheet_fraction"] = fractions[2]

    extinction_features = {
        "molar_extinction_coefficient_reduced_cysteines",
        "molar_extinction_coefficient_oxidized_cysteines",
    }
    if feature_set & extinction_features:
        coefficients = protein_analysis.molar_extinction_coefficient()
        if "molar_extinction_coefficient_reduced_cysteines" in feature_set:
            computed_features["molar_extinction_coefficient_reduced_cysteines"] = coefficients[0]
        if "molar_extinction_coefficient_oxidized_cysteines" in feature_set:
            computed_features["molar_extinction_coefficient_oxidized_cysteines"] = coefficients[1]

    return computed_features


def _validate_features_have_scaler_params(feature_list: list[str], is_peptide: bool = False) -> None:
    """Validate that all requested features have normalization scaler parameters."""
    if is_peptide:
        available_features = set(BIOPYTHON_PEPTIDE_SCALER_PARAMS.keys())
    else:
        available_features = set(BIOPYTHON_PROTEIN_SCALER_PARAMS.keys())

    missing_distributions = [name for name in feature_list if name not in available_features]

    if any(missing_distributions):
        raise RuntimeError(
            f"The following features do not have {'protein' if is_peptide else 'peptide'} scaler parameters available: {missing_distributions}. "
        )


def _validate_if_peptide_is_appropriate(sequence: str, is_peptide: bool) -> None:
    """Validate that the sequence length is appropriate for the given is_peptide."""
    if len(sequence) < PEPTIDE_WARNING_THRESHOLD and not is_peptide:
        logger.critical(
            f"Sequence length (={len(sequence)}) is less than the peptide warning threshold (={PEPTIDE_WARNING_THRESHOLD}), but is_peptide is set to False. "
            "Normalization may not be appropriate."
        )

    elif len(sequence) >= PEPTIDE_WARNING_THRESHOLD and is_peptide:
        logger.critical(
            f"Sequence length (={len(sequence)}) is greater than the peptide warning threshold (={PEPTIDE_WARNING_THRESHOLD}), but is_peptide is set to True. "
            "Normalization may not be appropriate."
        )


def _validate_feature_list(feature_list: list[str]) -> None:
    """Validate that the feature list is valid."""
    invalid_features = set(feature_list) - set(BIOPYTHON_FEATURES)
    if invalid_features:
        raise ValueError(f"Invalid features requested: {invalid_features}. Available features: {BIOPYTHON_FEATURES}")


def get_biopython_features(
    sequence: str,
    feature_list: Sequence[str] | None = None,
    return_as_tensor: bool = False,
    complex_separator: str | None = ".",
) -> dict[str, float] | torch.Tensor:
    """
    Extract BioPython features from a protein sequence or protein complex.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes) or protein complex with chains
        separated by the complex_separator.
    feature_list : Sequence[str] | None
        List of specific features to extract. If None, returns all features.
        Available features: sequence_length, molecular_weight, aromaticity_index,
        instability_index, isoelectric_point, alpha_helix_fraction, turn_structure_fraction,
        beta_sheet_fraction, molar_extinction_coefficient_reduced_cysteines,
        molar_extinction_coefficient_oxidized_cysteines, grand_average_hydropathy_index,
        net_charge_at_ph_6, net_charge_at_ph_7.
    return_as_tensor : bool
        If True, returns features as a tensor. If False, returns as a dictionary.
        Defaults to False.
    complex_separator : str | None
        Separator used to split protein complexes into individual chains. If None,
        treats the entire sequence as a single chain. Defaults to ".".

    Returns
    -------
    dict[str, float] | torch.Tensor
        Dictionary of BioPython features or tensor of feature values. For protein
        complexes, features are aggregated across chains using appropriate methods:
        - Extensive properties (sequence_length, molecular_weight, charges, extinction coefficients): summed
        - Intensive properties (all others): averaged
    """
    if feature_list is None:
        feature_list = list(BIOPYTHON_FEATURES)

    _validate_feature_list(feature_list)

    # Check if this is a protein complex
    if complex_separator and complex_separator in sequence:
        chains = _split_protein_complex(sequence, complex_separator)

        chain_features = []
        for chain in chains:
            chain_feature_dict = _compute_single_chain_features(chain, feature_list)
            chain_features.append(chain_feature_dict)

        computed_features = _aggregate_features(chain_features, feature_list)
    else:
        # Single chain processing
        computed_features = _compute_single_chain_features(sequence, feature_list)

    if not return_as_tensor:
        return {name: float(computed_features[name]) for name in feature_list}

    # Ensure consistent ordering for tensor output
    feature_values = [computed_features[name] for name in feature_list]

    return torch.tensor(feature_values, dtype=torch.float32)


def get_standardized_biopython_features(
    sequence: str,
    feature_list: Sequence[str] | None = None,
    return_as_tensor: bool = False,
    complex_separator: str | None = ".",
    is_peptide: bool = False,
) -> dict[str, float] | torch.Tensor:
    """Get standard-scaled BioPython features for a protein/peptide sequence.


    Protein standard-scaling parameters were obtained by fitting a StandardScaler to the protein features of the AMPLIFY dataset
    and peptide parameters were obtained from PeptideAtlas.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes) or protein complex with chains
        separated by the complex_separator.
    feature_list : Sequence[str] | None
        List of specific features to extract. If None, returns all features.
    return_as_tensor : bool
        If True, returns features as a tensor. If False, returns as a dictionary.
        Defaults to False.
    complex_separator : str | None
        Separator used to split protein complexes into individual chains. If None,
        treats the entire sequence as a single chain. Defaults to ".".
    is_peptide : bool
        If True, the sequence is a peptide. If False, the sequence is a protein.
        Defaults to False.

    Returns
    -------
    dict[str, float] | torch.Tensor
        Dictionary of standard-scaled BioPython features or tensor of feature values.
    """
    feature_list = list(feature_list) if feature_list is not None else list(BIOPYTHON_FEATURES)
    _validate_features_have_scaler_params(feature_list, is_peptide)
    _validate_if_peptide_is_appropriate(sequence, is_peptide)

    features = get_biopython_features(
        sequence, feature_list, return_as_tensor=False, complex_separator=complex_separator
    )

    scaler_parameters = BIOPYTHON_PEPTIDE_SCALER_PARAMS if is_peptide else BIOPYTHON_PROTEIN_SCALER_PARAMS

    scaled_features = {}

    for feature_name, x in features.items():
        scaler_params = scaler_parameters[feature_name]
        x_scaled = (x - scaler_params["mean"]) / scaler_params["scale"]

        scaled_features[feature_name] = x_scaled

    if return_as_tensor:
        return torch.tensor(list(scaled_features.values()), dtype=torch.float32)
    else:
        return scaled_features
