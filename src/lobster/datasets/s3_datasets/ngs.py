from collections.abc import Callable

from lobster.constants import Modality, Split

from .base import UMEStreamingDataset


class NGS(UMEStreamingDataset):
    MODALITY = Modality.AMINO_ACID
    SEQUENCE_KEY = "aa_seq"

    TRAIN_SIZE = 360_717_560
    VAL_SIZE = 125_006_827  # 686_094
    TEST_SIZE = 125_006_827

    LIMIT_VAL_SIZE = 20_000

    SPLITS = {
        Split.TRAIN: "s3://prescient-collaboration-dev/ngs/antibody_seq_sharded/train",
        Split.VALIDATION: "s3://prescient-collaboration-dev/ngs/antibody_seq_sharded/test",  # TODO change to val
        Split.TEST: "s3://prescient-collaboration-dev/ngs/antibody_seq_sharded/test",
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
            use_optimized=False,
            tokenize=True,
            max_length=max_length,
            **kwargs,
        )
