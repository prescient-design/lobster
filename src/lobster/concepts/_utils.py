alphabet = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    ".",
    "-",
]


supported_biopython_concepts = [
    "molecular_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy",
    "charge_at_pH6",
    "charge_at_pH7",
    "helix_fraction",
    "turn_structure_fraction",
    "sheet_structure_fraction",
    "molar_extinction_coefficient_reduced",
    "molar_extinction_coefficient_oxidized",
    "avg_hydrophilicity",
    "avg_surface_accessibility",
]


def normalize(value, min_value, max_value):
    if min_value is not None and max_value is not None:
        if (max_value - min_value) == 0:
            return value
        else:
            normalized_value = (value - min_value) / (max_value - min_value)
            return normalized_value
    else:
        return value
