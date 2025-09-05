from collections.abc import Callable

from lobster.constants import Modality, Split, S3_BUCKET

from .base import UMEStreamingDataset


class Calm(UMEStreamingDataset):
    MODALITY = Modality.NUCLEOTIDE
    SEQUENCE_KEY = "sequence"

    TRAIN_SIZE = 8_600_758
    VAL_SIZE = 87_758
    LIMIT_VAL_SIZE = 20_000
    NUM_TRAIN_TOKENS = 9_635_735_915

    SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/calm/processed/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/calm/processed/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/calm/processed/test",
    }
    OPTIMIZED_SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/calm/lightning-optimized/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/calm/lightning-optimized/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/calm/lightning-optimized/test",
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
