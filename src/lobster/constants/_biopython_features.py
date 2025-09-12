import numpy as np

BIOPYTHON_FEATURES = [
    "sequence_length",
    "molecular_weight",
    "aromaticity_index",
    "instability_index",
    "isoelectric_point",
    "alpha_helix_fraction",
    "turn_structure_fraction",
    "beta_sheet_fraction",
    "molar_extinction_coefficient_reduced_cysteines",
    "molar_extinction_coefficient_oxidized_cysteines",
    "grand_average_hydropathy_index",
    "net_charge_at_ph_6",
    "net_charge_at_ph_7",
]

# For complexes (or more than one chain): some can be summed, others averaged
BIOPYTHON_FEATURE_AGGREGATION_METHODS = {
    "sequence_length": sum,
    "molecular_weight": sum,
    "net_charge_at_ph_6": sum,
    "net_charge_at_ph_7": sum,
    "molar_extinction_coefficient_reduced_cysteines": sum,
    "molar_extinction_coefficient_oxidized_cysteines": sum,
    "aromaticity_index": np.nanmean,
    "instability_index": np.nanmean,
    "isoelectric_point": np.nanmean,
    "alpha_helix_fraction": np.nanmean,
    "turn_structure_fraction": np.nanmean,
    "beta_sheet_fraction": np.nanmean,
    "grand_average_hydropathy_index": np.nanmean,
}

PEPTIDE_WARNING_THRESHOLD = 50

# Obtained by fitting a StandardScaler to the computed BioPython features of the PeptideAtlas dataset
# Subset N=1,000,000, split=train, seed=0
BIOPYTHON_PEPTIDE_SCALER_PARAMS = {
    "alpha_helix_fraction": {
        "mean": 0.3190,
        "scale": 0.1436,
    },
    "net_charge_at_ph_7": {
        "mean": -0.9820,
        "scale": 2.1496,
    },
    "molar_extinction_coefficient_reduced_cysteines": {
        "mean": 1583.3512,
        "scale": 2571.6191,
    },
    "sequence_length": {
        "mean": 17.0051,
        "scale": 7.7124,
    },
    "aromaticity_index": {
        "mean": 0.0746,
        "scale": 0.0746,
    },
    "turn_structure_fraction": {
        "mean": 0.3076,
        "scale": 0.1354,
    },
    "molar_extinction_coefficient_oxidized_cysteines": {
        "mean": 1589.6785,
        "scale": 2572.9946,
    },
    "net_charge_at_ph_6": {
        "mean": -0.5155,
        "scale": 2.1259,
    },
    "grand_average_hydropathy_index": {
        "mean": -0.4819,
        "scale": 0.7528,
    },
    "beta_sheet_fraction": {
        "mean": 0.3245,
        "scale": 0.1233,
    },
    "molecular_weight": {
        "mean": 1885.8786,
        "scale": 817.9611,
    },
    "instability_index": {
        "mean": 43.3375,
        "scale": 34.6979,
    },
    "isoelectric_point": {
        "mean": 6.2018,
        "scale": 2.1132,
    },
}

# Obtained by fitting a StandardScaler to the computed BioPython features of the AMPLIFY dataset
# Subset N=1,000,000, split=train, seed=0
BIOPYTHON_PROTEIN_SCALER_PARAMS = {
    "turn_structure_fraction": {
        "mean": 0.2698,
        "scale": 0.0432,
    },
    "grand_average_hydropathy_index": {
        "mean": -0.0834,
        "scale": 0.4197,
    },
    "molar_extinction_coefficient_oxidized_cysteines": {
        "mean": 37048.6492,
        "scale": 34484.0984,
    },
    "net_charge_at_ph_7": {
        "mean": -3.3436,
        "scale": 12.9113,
    },
    "aromaticity_index": {
        "mean": 0.0805,
        "scale": 0.0324,
    },
    "molecular_weight": {
        "mean": 34719.3337,
        "scale": 26022.2456,
    },
    "sequence_length": {
        "mean": 316.0182,
        "scale": 238.6025,
    },
    "isoelectric_point": {
        "mean": 6.9221,
        "scale": 1.9587,
    },
    "net_charge_at_ph_6": {
        "mean": 0.3938,
        "scale": 12.1318,
    },
    "molar_extinction_coefficient_reduced_cysteines": {
        "mean": 36892.3279,
        "scale": 34401.3325,
    },
    "instability_index": {
        "mean": 37.6250,
        "scale": 11.1332,
    },
    "beta_sheet_fraction": {
        "mean": 0.3668,
        "scale": 0.0592,
    },
    "alpha_helix_fraction": {
        "mean": 0.3377,
        "scale": 0.0496,
    },
}
