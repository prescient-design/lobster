import logging
from collections.abc import Callable

from litdata import StreamingDataset

from lobster.constants import Split, S3_BUCKET

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

    def __next__(self):
        item = StreamingDataset.__next__(self)

        sequence1 = item.pop(self.SEQUENCE_KEY_1)
        sequence2 = item.pop(self.SEQUENCE_KEY_2)

        modality1 = item.pop(self.MODALITY_KEY_1)
        modality2 = item.pop(self.MODALITY_KEY_2)

        inputs = (sequence1, sequence2)
        modality = (modality1, modality2)

        input_ids, attention_mask, modality = self._tokenize_pair(sequence=inputs, modalities=modality)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence": inputs,
            "modality": modality,
            "dataset": self.__class__.__name__,
        }
