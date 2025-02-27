import logging
import warnings
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, List, Sequence, Tuple, Union

from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset, get_worker_info

from lobster.transforms import Transform

from ._utils import detect_distributed_environment

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
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        keys: Sequence[str] | None = None,
        split: str = "train",
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        seed: int = 0,
        download: bool = False,
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

        # Detect distributed environment
        self.distributed, self.rank, self.world_size = detect_distributed_environment()

        logging.info(f"Using distributed training: {self.distributed}")

        self.dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=not self.download,
            cache_dir=self.root,
            **self.kwargs,
        )

        if not isinstance(self.dataset, HFIterableDataset):
            self.dataset = self.dataset.to_iterable_dataset(num_shards=64)  # MANUAL

        if self.distributed:
            try:
                self.dataset = split_dataset_by_node(self.dataset, rank=self.rank, world_size=self.world_size)
                logging.info(f"Split dataset for node {self.rank} with world size {self.world_size}")
            except Exception as e:
                warnings.warn(
                    f"Failed to set up distributed dataset: {e}. Falling back to non-distributed mode.", stacklevel=2
                )
                self.distributed = False

    def _passes_type_check(self, sample: tuple[Any]) -> bool:
        """Implement a type check for the sample. Used for filtering out unwanted samples,
        such as those with missing data."""
        raise NotImplementedError

    def _process_sample(self, sample: tuple[Any]) -> Any:
        """Very simple processing of the sample. Anything that needs more complex
        processing should go into a transform."""
        return sample

    def __iter__(self) -> Iterator[Union[Tuple[str, ...], str]]:
        # Handle worker sharding if DataLoader uses multiple workers
        if (worker_info := get_worker_info()) is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Get the correct shard
        dataset = self.dataset.shard(num_shards=num_workers, index=worker_id)

        # Process samples from the dataset
        for sample in dataset:
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
