from Bio.SeqUtils.ProtParam import ProteinAnalysis
import torch
from collections.abc import Sequence

from lobster.constants import BIOPYTHON_FEATURES


def get_biopython_features(
    sequence: str, feature_list: Sequence[str] | None = None, return_as_tensor: bool = False
) -> dict[str, float] | torch.Tensor:
    """
    Extract BioPython features from a protein sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes).
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

    Returns
    -------
    dict[str, float] | torch.Tensor
        Dictionary of BioPython features or tensor of feature values.
    """
    if feature_list is None:
        feature_list = list(BIOPYTHON_FEATURES)

    # Validate feature list
    invalid_features = set(feature_list) - BIOPYTHON_FEATURES
    if invalid_features:
        raise ValueError(f"Invalid features requested: {invalid_features}. Available features: {BIOPYTHON_FEATURES}")

    feature_set = set(feature_list)

    protein_analysis = ProteinAnalysis(sequence)
    computed_features = {}

    # Sequence length (no ProteinAnalysis needed)
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

    if return_as_tensor:
        # Ensure consistent ordering for tensor output
        if feature_list is None:
            feature_list = sorted(list(BIOPYTHON_FEATURES))
        else:
            feature_list = sorted(feature_list)

        feature_values = [computed_features[feature_name] for feature_name in feature_list]
        return torch.tensor(feature_values, dtype=torch.float32)

    return computed_features
