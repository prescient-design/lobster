from enum import StrEnum
from typing import Literal

# TODO: currently, these will work for internal users
# External users should use HuggingFace to download the models
UME_CHECKPOINT_DICT_S3_BUCKET = "prescient-lobster"
UME_CHECKPOINT_DICT_S3_KEY = "ume/checkpoints.json"
UME_CHECKPOINT_DICT_S3_URI = f"s3://{UME_CHECKPOINT_DICT_S3_BUCKET}/{UME_CHECKPOINT_DICT_S3_KEY}"


UME_MODEL_VERSION_TYPES = Literal[
    "ume-large-base-870M",
    "ume-medium-base-480M",
    "ume-small-base-90M",
    "ume-mini-base-12M",
]


class UMEModelVersion(StrEnum):
    """Enum for UME model versions."""

    LARGE = "ume-large-base-870M"
    MEDIUM = "ume-medium-base-480M"
    SMALL = "ume-small-base-90M"
    MINI = "ume-mini-base-12M"
