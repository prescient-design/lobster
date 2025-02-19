from enum import Enum

# TODO - add to __init__

CALM_DATA_GITHUB_URL = "https://raw.githubusercontent.com/oxpig/CaLM/main/data"

class Species(str, Enum):
    ATHALIANA = "athaliana"
    DMELANOGASTER = "dmelanogaster"
    ECOLI = "ecoli"
    HSAPIENS = "hsapiens"
    HVOLCANII = "hvolcanii"
    PPASTORIS = "ppastoris"
    SCEREVISIAE = "scerevisiae"

class Task(Enum):
    MELTOME = "meltome"
    SOLUBILITY = "solubility"
    LOCALIZATION = "localization"
    PROTEIN_ABUNDANCE = "protein_abundance"
    TRANSCRIPT_ABUNDANCE = "transcript_abundance"
    SPECIES = "species"
    FUNCTION_BP = "function_bp"  # Separate task for each function type 
    FUNCTION_CC = "function_cc"
    FUNCTION_MF = "function_mf"

FUNCTION_ZENODO_BASE_URL = "https://zenodo.org/records/14890750/files"
FUNCTION_HASHES = {
    "function_bp": "md5:898265de59ba1ac97270bffc3621f334",
    "function_cc": "md5:a6af91fe40e523c9adf47e6abd98d9c6",
    "function_mf": "md5:cafe14db5dda19837bae536399e47e35"
}


TASK_SPECIES = {
    Task.PROTEIN_ABUNDANCE: [
        Species.ATHALIANA,
        Species.DMELANOGASTER,
        Species.ECOLI,
        Species.HSAPIENS,
        Species.SCEREVISIAE,
    ],
    Task.TRANSCRIPT_ABUNDANCE: [
        Species.ATHALIANA,
        Species.DMELANOGASTER,
        Species.ECOLI,
        Species.HSAPIENS,
        Species.HVOLCANII,
        Species.PPASTORIS,
        Species.SCEREVISIAE,
    ],
    Task.SPECIES: [
        Species.ATHALIANA,
        Species.DMELANOGASTER,
        Species.ECOLI,
        Species.HSAPIENS,
        Species.HVOLCANII,
        Species.PPASTORIS,
        Species.SCEREVISIAE,
    ]
}

