from typing import Literal
# TODO: currently, these will work for internal users
# Support for external users will be added soon

UME_CHECKPOINT_DICT_S3_BUCKET = "prescient-lobster"
UME_CHECKPOINT_DICT_S3_KEY = "ume/checkpoints.json"
UME_CHECKPOINT_DICT_S3_URI = f"s3://{UME_CHECKPOINT_DICT_S3_BUCKET}/{UME_CHECKPOINT_DICT_S3_KEY}"

UME_MODEL_VERSIONS = [
    "ume-large-base-740M",
    "ume-medium-base-480M",
    "ume-small-base-90M",
    "ume-mini-base-12M",
]
UME_MODEL_VERSIONS_TYPES = Literal[
    "ume-large-base-740M",
    "ume-medium-base-480M",
    "ume-small-base-90M",
    "ume-mini-base-12M",
]
