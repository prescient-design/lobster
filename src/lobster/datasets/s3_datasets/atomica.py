import logging
from collections.abc import Callable
from collections.abc import Sequence

from litdata import StreamingDataset

from lobster.constants import S3_BUCKET, Split
from lobster.constants._modality import Modality

from .base import UMEStreamingDataset

logger = logging.getLogger(__name__)


class Atomica(UMEStreamingDataset):
    SEQUENCE_KEY_1 = "sequence1"
    SEQUENCE_KEY_2 = "sequence2"

    MODALITY_KEY_1 = "modality1"
    MODALITY_KEY_2 = "modality2"

    TRAIN_SIZE = 309867
    VAL_SIZE = 16309

    SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/atomica/processed/split=train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/atomica/processed/split=val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/atomica/processed/split=test",
    }

    OPTIMIZED_SPLITS = {
        Split.TRAIN: f"s3://{S3_BUCKET}/ume/datasets/atomica/lightning-optimized/train",
        Split.VALIDATION: f"s3://{S3_BUCKET}/ume/datasets/atomica/lightning-optimized/val",
        Split.TEST: f"s3://{S3_BUCKET}/ume/datasets/atomica/lightning-optimized/test",
    }

    def __init__(
        self,
        split: Split,
        max_length: int,
        seed: int = 0,
        cache_dir: str | None = None,
        transform_fn: Callable | None = None,
        use_optimized: bool = False,
        restrict_modalities: Sequence[Modality] | None = None,
        **kwargs,
    ):
        super().__init__(
            split=split,
            seed=seed,
            cache_dir=cache_dir,
            transform_fn=transform_fn,
            use_optimized=use_optimized,
            tokenize=True,
            max_length=max_length,
            **kwargs,
        )

        self.restrict_modalities = set(
            Modality(modality) if isinstance(modality, str) else modality for modality in restrict_modalities or []
        )

    def __next__(self):
        item = StreamingDataset.__next__(self)

        sequence1 = item.pop(self.SEQUENCE_KEY_1)
        sequence2 = item.pop(self.SEQUENCE_KEY_2)

        modality1 = item.pop(self.MODALITY_KEY_1)
        modality2 = item.pop(self.MODALITY_KEY_2)

        if self.restrict_modalities and any(
            modality not in self.restrict_modalities for modality in [modality1, modality2]
        ):
            return self.__next__()

        encoded1 = self.tokenizer_registry[modality1](sequence1)
        encoded2 = self.tokenizer_registry[modality2](sequence2)

        return {
            "input_ids1": encoded1["input_ids"],
            "attention_mask1": encoded1["attention_mask"],
            "input_ids2": encoded2["input_ids"],
            "attention_mask2": encoded2["attention_mask"],
            "modality1": modality1,
            "modality2": modality2,
            "sequence1": sequence1,
            "sequence2": sequence2,
            "dataset": self.__class__.__name__,
        }
