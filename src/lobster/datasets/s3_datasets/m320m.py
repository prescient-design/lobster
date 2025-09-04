from collections.abc import Callable

from lobster.constants import Modality, Split, S3_BUCKET

from .base import UMEStreamingDataset


class M320M(UMEStreamingDataset):
    MODALITY = Modality.SMILES
    SEQUENCE_KEY = "smiles"

    TRAIN_SIZE = 20_787_194
    VAL_SIZE = 212_078
    LIMIT_VAL_SIZE = 20_000
    NUM_TRAIN_TOKENS = 1_075_293_154

    SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/m320m/processed/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/m320m/processed/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/m320m/processed/test",
    }

    OPTIMIZED_SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/m320m/lightning-optimized/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/m320m/lightning-optimized/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/m320m/lightning-optimized/test",
    }

    def __init__(
        self,
        split: Split,
        seed: int = 0,
        cache_dir: str | None = None,
        transform_fn: Callable | None = None,
        max_length: int = 512,
        use_optimized: bool = False,
        **kwargs,
    ):
        super().__init__(
            split=split,
            seed=seed,
            cache_dir=cache_dir,
            transform_fn=transform_fn,
            tokenize=True,
            max_length=max_length,
            use_optimized=use_optimized,
            **kwargs,
        )
