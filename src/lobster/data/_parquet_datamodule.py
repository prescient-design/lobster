from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any, TypeVar

import logging
import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from lobster.transforms import Transform

from ._dataframe_dataset_in_memory import DataFrameDatasetInMemory

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ParquetLightningDataModule(L.LightningDataModule):
    """LightningDataModule for loading parquet files with pre-split train/val/test.

    This DataModule loads parquet files from S3 or local paths, creates
    DataFrameDatasetInMemory instances, and provides DataLoaders for training,
    validation, and testing.

    Parameters
    ----------
    train_path : str
        Path to training parquet file(s) (S3 or local)
    val_path : str
        Path to validation parquet file(s) (S3 or local)
    test_path : str
        Path to test parquet file(s) (S3 or local)
    sequence_column : str
        Column name containing protein sequences
    label_column : str
        Column name containing target labels
    transform_fn : Callable | Transform | None, default=None
        Transform to apply to sequences (e.g., AutoTokenizerTransform)
    target_transform_fn : Callable | Transform | None, default=None
        Transform to apply to labels (e.g., BinarizeLabelTransform for classification)
    batch_size : int, default=32
        Batch size for dataloaders
    shuffle : bool, default=True
        Whether to shuffle training data
    sampler : Iterable | Sampler | None, default=None
        Optional sampler for dataloaders
    batch_sampler : Iterable[Sequence] | Sampler[Sequence] | None, default=None
        Optional batch sampler
    num_workers : int, default=4
        Number of worker processes for data loading
    collate_fn : Callable[[list[T]], Any] | None, default=None
        Custom collate function. If None, uses default for ESM tokenized sequences
    pin_memory : bool, default=True
        Whether to use pinned memory for GPU transfer
    drop_last : bool, default=False
        Whether to drop last incomplete batch
    label_dtype : torch.dtype | str, default=torch.float32
        Data type for labels (torch.float32 or "float32" for regression,
        torch.long or "long" for classification)
    seed : int, default=42
        Random seed for debug sampling

    Examples
    --------
    >>> from lobster.data import ParquetLightningDataModule
    >>> from lobster.transforms import AutoTokenizerTransform
    >>>
    >>> transform = AutoTokenizerTransform(
    ...     pretrained_model_name_or_path="facebook/esm2_t33_650M_UR50D",
    ...     max_length=512,
    ...     padding="max_length",
    ...     truncation=True,
    ... )
    >>>
    >>> # Optional: Add target transform for classification
    >>> from lobster.transforms import BinarizeLabelTransform
    >>> target_transform = BinarizeLabelTransform(threshold=1.0)
    >>>
    >>> datamodule = ParquetLightningDataModule(
    ...     train_path="s3://bucket/train/",
    ...     val_path="s3://bucket/val/",
    ...     test_path="s3://bucket/test/",
    ...     sequence_column="sequence",
    ...     label_column="target",
    ...     transform_fn=transform,
    ...     target_transform_fn=target_transform,  # Optional for classification
    ...     batch_size=32,
    ...     label_dtype=torch.long,  # Use long for classification
    ... )
    """

    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        sequence_column: str,
        label_column: str,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
        batch_size: int = 32,
        shuffle: bool = True,
        sampler: Iterable | Sampler | None = None,
        batch_sampler: Iterable[Sequence] | Sampler[Sequence] | None = None,
        num_workers: int = 4,
        collate_fn: Callable[[list[T]], Any] | None = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        label_dtype: torch.dtype | str = torch.float32,
        seed: int = 42,
        debug: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transform_fn", "target_transform_fn", "collate_fn"])

        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = seed
        self.debug = debug

        if isinstance(label_dtype, str):
            self.label_dtype = getattr(torch, label_dtype)
        else:
            self.label_dtype = label_dtype

        if collate_fn is None:
            self.collate_fn = partial(_default_collate_fn, label_dtype=self.label_dtype)
        else:
            self.collate_fn = collate_fn

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Load parquet files (called on single GPU/process)."""
        pass

    def setup(self, stage: str = "fit"):
        """Load data and create datasets (called on each GPU/process).

        Parameters
        ----------
        stage : str, default="fit"
            Stage of training (fit, test, predict)
        """
        if stage == "fit" or stage is None:
            train_df = pd.read_parquet(self.train_path)
            train_df = train_df.dropna(subset=[self.sequence_column, self.label_column])
            logger.info(f"  Loaded {len(train_df):,} training samples")

            val_df = pd.read_parquet(self.val_path)
            val_df = val_df.dropna(subset=[self.sequence_column, self.label_column])

            test_df = pd.read_parquet(self.test_path)
            test_df = test_df.dropna(subset=[self.sequence_column, self.label_column])
            logger.info(f"  Loaded {len(test_df):,} test samples")

            if self.debug:
                min_samples = 100
                min_samples = min(min_samples, len(train_df), len(val_df), len(test_df))
                train_df = train_df.sample(n=min_samples, random_state=self.seed)
                val_df = val_df.sample(n=min_samples, random_state=self.seed)
                test_df = test_df.sample(n=min_samples, random_state=self.seed)
                logger.info(f"  Debug mode: Sampled {min_samples:,} of train, val, and test samples")

            self.train_dataset = DataFrameDatasetInMemory(
                data=train_df,
                columns=[self.sequence_column],
                target_columns=[self.label_column],
                transform_fn=self.transform_fn,
                target_transform_fn=self.target_transform_fn,
            )

            self.val_dataset = DataFrameDatasetInMemory(
                data=val_df,
                columns=[self.sequence_column],
                target_columns=[self.label_column],
                transform_fn=self.transform_fn,
                target_transform_fn=self.target_transform_fn,
            )

            self.test_dataset = DataFrameDatasetInMemory(
                data=test_df,
                columns=[self.sequence_column],
                target_columns=[self.label_column],
                transform_fn=self.transform_fn,
                target_transform_fn=self.target_transform_fn,
            )

        elif stage == "test":
            test_df = pd.read_parquet(self.test_path)
            test_df = test_df.dropna(subset=[self.sequence_column, self.label_column])
            logger.info(f"  Loaded {len(test_df):,} test samples")

            self.test_dataset = DataFrameDatasetInMemory(
                data=test_df,
                columns=[self.sequence_column],
                target_columns=[self.label_column],
                transform_fn=self.transform_fn,
                target_transform_fn=self.target_transform_fn,
            )

        elif stage == "predict":
            test_df = pd.read_parquet(self.test_path)
            test_df = test_df.dropna(subset=[self.sequence_column, self.label_column])
            logger.info(f"  Loaded {len(test_df):,} prediction samples")

            self.predict_dataset = DataFrameDatasetInMemory(
                data=test_df,
                columns=[self.sequence_column],
                target_columns=[self.label_column],
                transform_fn=self.transform_fn,
                target_transform_fn=self.target_transform_fn,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
        )


def _default_collate_fn(batch, label_dtype=torch.float32):
    """Default collate function for batching tokenized sequences.

    Parameters
    ----------
    batch : list of tuples
        List of (tokenized_dict, label) tuples
    label_dtype : torch.dtype, default=torch.float32
        Data type to cast labels to

    Returns
    -------
    dict
        Batched dictionary with input_ids, attention_mask, and labels
    """
    inputs, labels = zip(*batch)

    # AutoTokenizerTransform returns BatchEncoding, extract tensors
    input_ids = torch.stack([item["input_ids"].squeeze(0) for item in inputs])
    attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in inputs])
    labels = torch.stack([label for label in labels]).to(label_dtype)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
