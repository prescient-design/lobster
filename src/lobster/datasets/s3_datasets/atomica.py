import logging
from collections.abc import Callable
from collections.abc import Sequence

from litdata import StreamingDataset

from lobster.constants import S3_BUCKET, Split, Modality, to_modality

from .base import UMEStreamingDataset

logger = logging.getLogger(__name__)


class Atomica(UMEStreamingDataset):
    SEQUENCE_KEY_1 = "sequence1"
    SEQUENCE_KEY_2 = "sequence2"
    SEQUENCE_KEY_3 = "sequence3"

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

        self.restrict_modalities = set(to_modality(modality) for modality in restrict_modalities or [])
        self.num_skipped_items = 0

    def __next__(self):
        item = StreamingDataset.__next__(self)

        sequence1 = item.pop(self.SEQUENCE_KEY_1)
        sequence2 = item.pop(self.SEQUENCE_KEY_2)
        sequence3 = item.pop(self.SEQUENCE_KEY_3, None)

        if self._should_skip(sequence1, sequence2, sequence3):
            self.num_skipped_items += 1
            return self.__next__()

        modality1 = to_modality(item.pop(self.MODALITY_KEY_1))
        modality2 = to_modality(item.pop(self.MODALITY_KEY_2))

        if self.restrict_modalities and (
            modality1 not in self.restrict_modalities or modality2 not in self.restrict_modalities
        ):
            logger.debug(
                f"Skipping item with modalities {modality1} and {modality2} because they are not in {self.restrict_modalities}"
            )
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

    def _should_skip(self, sequence1: str | None, sequence2: str | None, sequence3: str | None) -> bool:
        if sequence1 is None or sequence2 is None:
            logger.debug("Skipping item because sequence1 or sequence2 is None")
            return True

        if sequence3 is not None:
            logger.debug(
                "Skipping item because it has 2+ sequences (and we can't guarantee that interaction is between sequence1 and sequence2)"
            )
            return True

        if sequence1 == sequence2:
            logger.debug(f"Skipping item because sequences are the same: {sequence1}")
            return True

        if len(sequence1) > self.max_length or len(sequence2) > self.max_length:
            logger.debug(
                f"Skipping item because one or both sequences are too long: {len(sequence1)} and {len(sequence2)} (max length: {self.max_length})"
            )
            return True

        logger.debug(f"Item passed all checks: {sequence1} and {sequence2}")
        return False
