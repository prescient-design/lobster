from Bio.SeqUtils.ProtParam import ProteinAnalysis
import torch

from lobster.constants import BIOPYTHON_FEATURES


def get_biopython_features(sequence: str, feature_list: list[str] | None = None) -> dict[str, float]:
    """
    Extract BioPython features from a protein sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes).
    feature_list : list[str] | None
        List of specific features to extract. If None, returns all features.
        Available features: sequence_length, molecular_weight, aromaticity_index,
        instability_index, isoelectric_point, alpha_helix_fraction, turn_structure_fraction,
        beta_sheet_fraction, molar_extinction_coefficient_reduced_cysteines,
        molar_extinction_coefficient_oxidized_cysteines, grand_average_hydropathy_index,
        net_charge_at_ph_6, net_charge_at_ph_7.

    Returns
    -------
    dict[str, float]
        Dictionary of BioPython features.
    """
    if feature_list is None:
        feature_list = list(BIOPYTHON_FEATURES)

    # Validate feature list
    invalid_features = set(feature_list) - BIOPYTHON_FEATURES

    if invalid_features:
        raise ValueError(f"Invalid features requested: {invalid_features}. Available features: {BIOPYTHON_FEATURES}")

    protein_analysis = ProteinAnalysis(sequence)

    # Pre-compute expensive operations only if needed
    molar_extinction_coefficients = None
    secondary_structure_fractions = None

    if any(
        feature in feature_list
        for feature in [
            "molar_extinction_coefficient_reduced_cysteines",
            "molar_extinction_coefficient_oxidized_cysteines",
        ]
    ):
        molar_extinction_coefficients = protein_analysis.molar_extinction_coefficient()

    if any(
        feature in feature_list
        for feature in ["alpha_helix_fraction", "turn_structure_fraction", "beta_sheet_fraction"]
    ):
        secondary_structure_fractions = protein_analysis.secondary_structure_fraction()

    # Feature computation mapping
    feature_computation_map = {
        "sequence_length": lambda: len(sequence),
        "molecular_weight": lambda: protein_analysis.molecular_weight(),
        "aromaticity_index": lambda: protein_analysis.aromaticity(),
        "instability_index": lambda: protein_analysis.instability_index(),
        "isoelectric_point": lambda: protein_analysis.isoelectric_point(),
        "alpha_helix_fraction": lambda: secondary_structure_fractions[0]
        if secondary_structure_fractions is not None
        else protein_analysis.secondary_structure_fraction()[0],
        "turn_structure_fraction": lambda: secondary_structure_fractions[1]
        if secondary_structure_fractions is not None
        else protein_analysis.secondary_structure_fraction()[1],
        "beta_sheet_fraction": lambda: secondary_structure_fractions[2]
        if secondary_structure_fractions is not None
        else protein_analysis.secondary_structure_fraction()[2],
        "molar_extinction_coefficient_reduced_cysteines": lambda: molar_extinction_coefficients[0]
        if molar_extinction_coefficients is not None
        else protein_analysis.molar_extinction_coefficient()[0],
        "molar_extinction_coefficient_oxidized_cysteines": lambda: molar_extinction_coefficients[1]
        if molar_extinction_coefficients is not None
        else protein_analysis.molar_extinction_coefficient()[1],
        "grand_average_hydropathy_index": lambda: protein_analysis.gravy(),
        "net_charge_at_ph_6": lambda: protein_analysis.charge_at_pH(6),
        "net_charge_at_ph_7": lambda: protein_analysis.charge_at_pH(7),
    }

    # Compute only requested features
    computed_features = {}
    for feature_name in feature_list:
        computed_features[feature_name] = feature_computation_map[feature_name]()

    return computed_features


def protein_to_biopython_features_tensor(sequence: str, feature_list: list[str] | None = None) -> torch.Tensor:
    """
    Convert protein sequence to BioPython features as a tensor.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter codes).
    feature_list : list[str] | None
        List of specific features to extract. If None, returns all features.

    Returns
    -------
    torch.Tensor
        Tensor of BioPython features in the order specified by feature_list
        (or default order if feature_list is None).
    """
    if feature_list is None:
        feature_list = sorted(list(BIOPYTHON_FEATURES))  # Consistent ordering

    features_dictionary = get_biopython_features(sequence, feature_list)
    feature_values = [features_dictionary[feature_name] for feature_name in feature_list]

    return torch.tensor(feature_values, dtype=torch.float32)
