from collections.abc import Callable

from lobster.constants import Modality, Split, S3_BUCKET

from .base import S3StreamingDataset


class ZINC(S3StreamingDataset):
    MODALITY = Modality.SMILES
    SEQUENCE_KEY = "smiles"

    TRAIN_SIZE = 588_651_680
    VAL_SIZE = 6_006_398
    LIMIT_VAL_SIZE = 5000
    NUM_TRAIN_TOKENS = 23_554_412_701

    SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/zinc/processed/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/zinc/processed/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/zinc/processed/test",
    }
    OPTIMIZED_SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/zinc/lightning-optimized/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/zinc/lightning-optimized/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/zinc/lightning-optimized/test",
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
