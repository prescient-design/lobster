from enum import Enum

# TODO - add to __init__

CALM_DATA_GITHUB_URL = "https://raw.githubusercontent.com/oxpig/CaLM/main/data"
FUNCTION_ZENODO_BASE_URL = "https://zenodo.org/records/14890750/files" # Gene Ontology datasets processed & uploaded on Zenodo
FILE_HASHES = {    
    "meltome.csv": "sha256:699074debc9e3d66e0c084bca594ce81d26b3126b645d43b0597dbe466153ad4",
    "solubility.csv": "sha256:94b351d0f36b490423b3e80b2ff0ea5114423165948183157cf63d4f57c08d38",
    "localization.csv": "sha256:efedb7c394b96f4569c72d03eac54ca5a3b4a24e15c9c24d9429f9b1a4e29320",
    
    # Function prediction tasks
    "calm_GO_bp_middle_normal.parquet": "md5:898265de59ba1ac97270bffc3621f334",
    "calm_GO_cc_middle_normal.parquet": "md5:a6af91fe40e523c9adf47e6abd98d9c6",
    "calm_GO_mf_middle_normal.parquet": "md5:cafe14db5dda19837bae536399e47e35",

    # Protein abundance
    "protein_abundance_athaliana.csv": "sha256:83f8d995ee3a0ff6f1ed4e74a9cb891546e2edb6e86334fef3b6901a0039b118",
    "protein_abundance_dmelanogaster.csv": "sha256:6f9541d38217f71a4f9addec4ad567d60eee4e4cebb1d16775909b88e1775da4",
    "protein_abundance_ecoli.csv": "sha256:a6a8f91901a4ea4da62931d1e7c91b3a6aa72e4d6c6a83a6612c0988e94421fb",
    "protein_abundance_hsapiens.csv": "sha256:94ded0486f2f64575bd2d8f2a3e00611a6e8b28b691d0f367ca9210058771a23",
    "protein_abundance_scerevisiae.csv": "sha256:0ce0b6a5b0400c3cc1c568f6c5478a974e80aaecbab93299f81bb94eb2373076",
    
    # Transcript abundance
    "transcript_abundance_athaliana.csv": "sha256:de7a6f57bcfbb60445d17b8461a8a3ea8353942e129f08ac2c6de5874cd6c139",
    "transcript_abundance_dmelanogaster.csv": "sha256:0124d101da004e7a66f4303ff097da39d5e4dd474548fa57e2f9fa7231544c32",
    "transcript_abundance_ecoli.csv": "sha256:5e480d961c8b9043f6039211241ecf904b188214e9951352a9b2fc3d6a630a59",
    "transcript_abundance_hsapiens.csv": "sha256:21b4b3f3f7267d28dbf6434090dfc0c58fde6f15393537d569f0b29e3eeec491",
    "transcript_abundance_hvolcanii.csv": "sha256:91782d2839f78b7c3a4c4d2c0f685605fa746e9b3936579fbd91ce875f9860aa",
    "transcript_abundance_ppastoris.csv": "sha256:4ebd4783e1b90e76e481c25bce801d4f6984f85d382f5d458d26f554e114798a",
    "transcript_abundance_scerevisiae.csv": "sha256:2e0f3b4c0cee77f47ab4009be881f671b709df368495f92dad66f34b2c88ac36"
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
    ]
}

