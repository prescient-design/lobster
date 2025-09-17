from collections.abc import Callable

from lobster.constants import S3_BUCKET, Modality, Split

from .base import UMEStreamingDataset


class PeptideAtlas(UMEStreamingDataset):
    MODALITY = Modality.AMINO_ACID
    SEQUENCE_KEY = "sequence"

    TRAIN_SIZE = 79_341_785
    VAL_SIZE = 12_641
    NUM_TRAIN_TOKENS = 79_341_785 * 10

    SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/peptide_atlas/processed/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/peptide_atlas/processed/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/peptide_atlas/processed/test",
    }
    OPTIMIZED_SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/peptide_atlas/lightning-optimized/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/peptide_atlas/lightning-optimized/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/peptide_atlas/lightning-optimized/test",
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
