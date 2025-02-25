from enum import Enum

CALM_TASKS = {
    "meltome": ("regression", None),  # (task_type, num_classes)
    "solubility": ("regression", None),
    "localization": ("multilabel", 10),  # 10 cellular locations
    "protein_abundance": ("regression", None),
    "transcript_abundance": ("regression", None),
    "function_bp": ("multilabel", 4),  # 4 GO terms
    "function_cc": ("multilabel", 4),
    "function_mf": ("multilabel", 4),
}


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
    FUNCTION_BP = "function_bp"
    FUNCTION_CC = "function_cc"
    FUNCTION_MF = "function_mf"


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
}
