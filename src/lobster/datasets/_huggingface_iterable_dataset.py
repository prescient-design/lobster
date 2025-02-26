import logging
import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset

from lobster.transforms import Transform

logger = logging.getLogger(__name__)


class HuggingFaceIterableDataset(IterableDataset):
    """Base Iterable Dataset class for loading streaming data from HuggingFace.

    This dataset handles proper worker distribution for DataLoader with multiple workers
    and automatically detects and configures for multi-node distributed training.
    """

    SUPPORTED_SPLITS: ClassVar[List[str]] = ["train"]  # Default, override in subclasses

    def __init__(
        self,
        dataset_name: str,
        root: Optional[Union[str, Path]] = None,
        *,
        transform: Optional[Union[Callable, Transform]] = None,
        keys: Sequence[str] | None = None,
        split: str = "train",
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        seed: Optional[int] = None,
        download: bool = False,
        distributed: Optional[bool] = None,
        **kwargs,
    ):
        """
        Initialize the base HuggingFace Iterable Dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset on HuggingFace.
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample.
        keys : Sequence of str or None, optional
            List of keys to be used from the dataset.
        split : str, optional
            Which split of the dataset to use.
        shuffle : bool, optional
            If True, shuffles the dataset.
        shuffle_buffer_size : int, optional
            Buffer size for shuffling streaming datasets.
        seed : int or None, optional
            Random seed for reproducible shuffling.
        download : bool, optional
            If True, download the dataset instead of streaming.
        distributed : bool or None, optional
            If True, force distributed mode. If False, disable distributed mode.
            If None (default), auto-detect distributed environment.
        kwargs : dict
            Additional keyword arguments to pass to the `load_dataset` function.
        """
        super().__init__()

        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(f"Split '{split}' is not supported.")

        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.download = download
        self.root = root
        self.kwargs = kwargs

        self.keys = list(keys) if keys is not None else None

        # Auto-detect distributed training environment if not explicitly set
        if distributed is None:
            # Check for PyTorch distributed
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

            # If PyTorch distributed is not initialized, check for common env variables
            if not distributed:
                distributed = all(var in os.environ for var in ["RANK", "WORLD_SIZE"])

        self.distributed = distributed

        # Get rank and world size for distributed setup
        if self.distributed:
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    self.rank = torch.distributed.get_rank()
                    self.world_size = torch.distributed.get_world_size()
                else:
                    self.rank = int(os.environ.get("RANK", 0))
                    self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            except Exception as e:
                warnings.warn(
                    f"Failed to get distributed info: {e}. Falling back to non-distributed setting", stacklevel=2
                )
                self.rank = 0
                self.world_size = 1
                self.distributed = False
        else:
            self.rank = 0
            self.world_size = 1

        # Create a worker-specific and node-specific seed if a seed is provided
        if self.seed is not None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0

            # Combine seed with rank and worker_id for unique seeds across nodes and workers
            self.actual_seed = self.seed + self.rank * 10000 + worker_id

            # Set the seed for reproducibility
            random.seed(self.actual_seed)

        # Load the dataset
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=not self.download,
            cache_dir=self.root,
            **self.kwargs,
        )

        # Make it an iterable dataset if it's not already (happens when download=True)
        if not isinstance(dataset, HFIterableDataset):
            dataset = dataset.to_iterable_dataset()

        # Apply shuffling before splitting for distributed training
        if self.shuffle:
            shuffle_seed = self.seed if self.seed is not None else 42
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=shuffle_seed)

        # Handle distributed training if enabled
        if self.distributed:
            try:
                # Split the dataset across nodes
                dataset = split_dataset_by_node(dataset, rank=self.rank, world_size=self.world_size)

                # Log information about the distributed setup
                logger.info(f"Dataset distributed: rank {self.rank}/{self.world_size}")

            except Exception as e:
                warnings.warn(
                    f"Failed to set up distributed dataset: {e}. Falling back to non-distributed mode.", stacklevel=2
                )
                self.distributed = False

        self.dataset = dataset

    def _passes_type_check(self, sample: tuple[Any]) -> bool:
        """Implement a type check for the sample.
        Used for filtering out unwanted samples, such as those with missing data."""
        raise NotImplementedError

    def _process_sample(self, sample: tuple[Any]) -> Any:
        return sample

    def __iter__(self) -> Iterator[Union[Tuple[str, ...], str]]:
        # Process examples from the dataset
        for sample in self.dataset:
            if self.keys is not None:
                sample = tuple(sample[k] for k in self.keys if k in sample)
            else:
                sample = tuple(sample.values())

            sample = self._process_sample(sample)

            if not self._passes_type_check(sample):
                continue

            if len(sample) == 1:
                sample = sample[0]

            if self.transform is not None:
                sample = self.transform(sample)

            yield sample
