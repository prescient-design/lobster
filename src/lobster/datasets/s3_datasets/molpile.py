from collections.abc import Callable

from lobster.constants import S3_BUCKET, Modality, Split

from .base import UMEStreamingDataset


class MolPILE(UMEStreamingDataset):
    MODALITY = Modality.SMILES
    SEQUENCE_KEY = "SMILES"

    TRAIN_SIZE = 223_000_000
    VAL_SIZE = 10_000

    SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/molpile/processed/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/molpile/processed/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/molpile/processed/test",
    }
    OPTIMIZED_SPLITS = {}

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
            use_optimized=False,  # ignore this parameter since we don't have optimized version
            **kwargs,
        )
