ESM_MODEL_NAMES = [
    "esm2_t6_8M_UR50D",  # 7.5M
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]

PMLM_MODEL_NAMES = [
    "MLM_small",  # 173K
    "MLM_med",  # 5M
    "MLM_large",  # 70M
    "MLM_cram_small",
    "MLM_cram_med",
    "MLM_cram_large",
    "MLM_cram_150M",
]

CLM_MODEL_NAMES = [
    "ProtGPT2",  # 708M trainable, non-embedding params
    "CLM_small",  # 7.2M
    "CLM_med",  # 22M
    "CLM_large",  # 176M
]

HEAVY_COLUMN = "fv_heavy"
LIGHT_COLUMN = "fv_light"
