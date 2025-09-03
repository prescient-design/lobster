from collections.abc import Callable

from lobster.constants import Modality, Split, S3_BUCKET
from lobster.transforms import ComposedModalityAwareTransform, ModalityAwareTransform

from .base import UMEStreamingDataset


class AMPLIFY(UMEStreamingDataset):
    MODALITY = Modality.AMINO_ACID
    SEQUENCE_KEY = "sequence"

    TRAIN_SIZE = 360_717_560
    VAL_SIZE = 3_644_010
    LIMIT_VAL_SIZE = 20_000
    NUM_TRAIN_TOKENS = 138_782_348_401

    SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/amplify/processed_v2/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/amplify/processed_v2/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/amplify/processed_v2/test",
    }

    OPTIMIZED_SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/amplify/lightning-optimized/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/amplify/lightning-optimized/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/amplify/lightning-optimized/test",
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
        # Create the sanitize transform
        sanitize = ModalityAwareTransform(
            lambda x: x.replace("|", "."), input_modality=self.MODALITY, output_modalities=(self.MODALITY,)
        )

        # If there's an additional transform, compose it with sanitize
        # using ComposedModalityAwareTransform to ensure we can preserve
        # the modality information
        if transform_fn is not None:
            transform = ComposedModalityAwareTransform(sanitize, transform_fn)
        else:
            transform = sanitize

        super().__init__(
            split=split,
            seed=seed,
            cache_dir=cache_dir,
            transform_fn=transform,
            use_optimized=use_optimized,
            tokenize=True,
            max_length=max_length,
            **kwargs,
        )
