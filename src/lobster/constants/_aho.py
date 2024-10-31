LENGTH_FV_LIGHT_AHO = 148
LENGTH_FV_HEAVY_AHO = 149

# ranges are not inclusive, i.e. [..), and zero-based as in python list indexing
CDR_RANGES_AHO = {
    "L1": (23, 42),
    "L2": (56, 72),
    "L3": (106, 138),
    "L4": (81, 89),
    "H1": (23, 42),
    "H2": (56, 69),
    "H3": (106, 138),
    "H4": (81, 89),
}

FR_RANGES_AHO = {
    "LFR1": (0, CDR_RANGES_AHO["L1"][0]),
    "LFR2": (CDR_RANGES_AHO["L1"][1], CDR_RANGES_AHO["L2"][0]),
    "LFR3a": (CDR_RANGES_AHO["L2"][1], CDR_RANGES_AHO["L4"][0]),
    "LFR3b": (CDR_RANGES_AHO["L4"][1], CDR_RANGES_AHO["L3"][0]),
    "LFR4": (CDR_RANGES_AHO["L3"][1], LENGTH_FV_LIGHT_AHO + 1),
    "HFR1": (0, CDR_RANGES_AHO["H1"][0]),
    "HFR2": (CDR_RANGES_AHO["H1"][1], CDR_RANGES_AHO["H2"][0]),
    "HFR3a": (CDR_RANGES_AHO["H2"][1], CDR_RANGES_AHO["H4"][0]),
    "HFR3b": (CDR_RANGES_AHO["H4"][1], CDR_RANGES_AHO["H3"][0]),
    "HFR4": (CDR_RANGES_AHO["H3"][1], LENGTH_FV_HEAVY_AHO + 1),
}

RANGES_AHO = {**FR_RANGES_AHO, **CDR_RANGES_AHO}

HUMAN_MOUSE_VERNIER_H = [
    "2",
    "24",
    "44",
    "46",
    "52",
    "54",
    "55",
    "56",
    "78",
    "80",
    "82",
    "84",
    "89",
    "105",
    "139",
    "141",
]
HUMAN_MOUSE_VERNIER_L = [
    "2",
    "4",
    "43",
    "44",
    "46",
    "51",
    "52",
    "54",
    "55",
    "56",
    "57",
    "74",
    "80",
    "82",
    "87",
    "71",
    "89",
    "139",
]
RABBIT_VERNIER_H = [
    "15",
    "19",
    "21",
    "23",
    "47",
    "54",
    "55",
    "56",
    "57",
    "79",
    "82",
    "83",
    "89",
    "90",
    "97",
    "99",
    "103",
    "105",
]
RABBIT_VERNIER_L = [
    "3",
    "7",
    "9",
    "14",
    "17",
    "18",
    "19",
    "22",
    "50",
    "51",
    "53",
    "79",
    "88",
]


VERNIER_ZONES = {
    "human": {"L": HUMAN_MOUSE_VERNIER_L, "H": HUMAN_MOUSE_VERNIER_H},
    "mouse": {"L": HUMAN_MOUSE_VERNIER_L, "H": HUMAN_MOUSE_VERNIER_H},
    "rabbit": {"L": RABBIT_VERNIER_H, "H": RABBIT_VERNIER_H},
}
