import logging
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info

from lobster.transforms import Transform

from ._distributed_environment_utils import detect_distributed_environment

logger = logging.getLogger(__name__)


class HuggingFaceIterableDataset(IterableDataset):
    """Base Iterable Dataset class for loading streaming data from HuggingFace.

    This dataset handles proper worker distribution for DataLoader with multiple workers
    and automatically detects and configures for multi-node distributed training.
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train"]  # Default, override in subclasses

    def __init__(
        self,
        dataset_name: str,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        keys: Sequence[str] | None = None,
        split: str = "train",
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        seed: int = 0,
        download: bool = False,
        limit: int | None = None,
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
        limit : int or None, optional
            Limit the number of samples to load.
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
        self.limit = limit
        self.kwargs = kwargs

        self.keys = list(keys) if keys is not None else None

        self.dataset: HFIterableDataset | HFDataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=not self.download,
            cache_dir=self.root,
            **self.kwargs,
        )

    def _passes_type_check(self, sample: tuple[Any]) -> bool:
        """Implement a type check for the sample. Used for filtering out unwanted samples,
        such as those with missing data."""
        raise NotImplementedError

    def _process_sample(self, sample: tuple[Any]) -> Any:
        """Very simple processing of the sample. Anything that needs more complex
        processing should go into a transform."""
        return sample

    def __iter__(self) -> Iterator[tuple[str, ...] | str]:
        # Detect distributed environment
        self.distributed, self.rank, self.world_size = detect_distributed_environment()

        # Get worker info
        if (worker_info := get_worker_info()) is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers = 1
            worker_id = 0

        # Calculate global worker ID and total workers
        global_worker_id = self.rank * num_workers + worker_id
        total_workers = self.world_size * num_workers

        # Split the dataset across nodes if in distributed mode
        if self.distributed:
            try:
                dataset = split_dataset_by_node(self.dataset, rank=self.rank, world_size=self.world_size)
            except Exception as e:
                warnings.warn(
                    f"Failed to set up distributed dataset: {e}. Falling back to non-distributed mode.", stacklevel=2
                )
                self.distributed = False
                dataset = self.dataset
        else:
            dataset = self.dataset

        # Convert to an iterable dataset if not already
        if not isinstance(dataset, HFIterableDataset):
            dataset = dataset.to_iterable_dataset(num_shards=num_workers)

        # Shuffle the dataset
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed)

        # Calculate per-worker limit
        if self.limit is not None:
            base_limit = self.limit // total_workers

            # Distribute the remainder samples
            # Workers with ID < extra get base_limit+1, others get base_limit
            extra = self.limit % total_workers
            worker_limit = base_limit + (1 if global_worker_id < extra else 0)

            logger.info(
                f"Worker {global_worker_id}/{total_workers} (rank={self.rank}, local_id={worker_id}) "
                f"will process up to {worker_limit} samples (from limit={self.limit})"
            )
        else:
            worker_limit = None

        # Process samples from the dataset
        count = 0
        for sample in dataset:
            # Check if we've reached this worker's limit
            if worker_limit is not None and count >= worker_limit:
                break

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

            count += 1
            yield sample
