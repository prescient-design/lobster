import random
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch.utils.data
from beignet.datasets import ChEMBLDataset
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.transforms import Transform

T = TypeVar("T")


class ChEMBLLightningDataModule(LightningDataModule):
    def __init__(
        self,
        root: str | Path = None,
        *,
        download: bool = False,
        transform_fn: Callable | Transform | None = None,
        lengths: Sequence[float] | None = (0.9, 0.05, 0.05),
        generator: Generator | None = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Iterable | Sampler | None = None,
        batch_sampler: Iterable[Sequence] | Sampler[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[T]], Any] | None = None,
        max_length: int = 512,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        """
        Initialize the ChEMBLLightningDataModule.

        Parameters
        ----------
        root : str or Path, optional
            Root directory where the dataset subdirectory exists or, if `download` is True,
            the directory where the dataset subdirectory will be created and the dataset downloaded.
        download : bool, optional
            If True, download the dataset to the `root` directory (default: False).
            If the dataset is already downloaded, it is not redownloaded.
        transform_fn : Callable or Transform or None, optional
            Function or transform to apply to the dataset for tokenization (default: None).
        lengths : Sequence[float] or None, optional
            Fractions of splits to generate (default: (0.9, 0.05, 0.05)).
        generator : Generator or None, optional
            Generator used for the random permutation (default: None).
        seed : int, optional
            Desired seed. Value must be within the inclusive range [-0x8000000000000000, 0xFFFFFFFFFFFFFFFF]
            (default: 0xDEADBEEF). Otherwise, a RuntimeError is raised. Negative inputs are remapped to positive
            values with the formula 0xFFFFFFFFFFFFFFFF + seed.
        batch_size : int, optional
            Samples per batch (default: 1).
        shuffle : bool, optional
            If True, reshuffle datasets at every epoch (default: True).
        sampler : Iterable or Sampler or None, optional
            Strategy to draw samples from the dataset (default: None). Can be any Iterable with __len__ implemented.
            If specified, `shuffle` must be False.
        batch_sampler : Iterable[Sequence] or Sampler[Sequence] or None, optional
            `sampler`, but returns a batch of indices (default: None). Mutually exclusive with `batch_size`,
            `shuffle`, `sampler`, and `drop_last`.
        num_workers : int, optional
            Subprocesses to use (default: 0). 0 means that the datasets will be loaded in the main process.
        collate_fn : Callable[[list[T]], Any] or None, optional
            Merges samples to form a mini-batch of Tensor(s) (default: None).
        max_length : int, optional
            Maximum length of the sequences (default: 512).
        pin_memory : bool, optional
            If True, Tensors are copied to the device's (e.g., CUDA) pinned memory before returning them (default: True).
        drop_last : bool, optional
            If True, drop the last incomplete batch, if the dataset size is not divisible by the batch size (default: False).
            If False and the size of dataset is not divisible by the batch size, then the last batch will be smaller.
        """
        super().__init__()

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._root = root
        self._download = download
        self._lengths = lengths
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._max_length = max_length
        self._shuffle = shuffle
        self._sampler = sampler
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._dataset = None
        self._transform_fn = transform_fn

    def prepare_data(self) -> None:
        dataset = ChEMBLDataset(
            root=self._root,
            download=self._download,
            transform=self._transform_fn,
        )
        self._dataset = dataset

    def setup(self, stage: str = "fit") -> None:
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        if self._dataset is None:
            self.prepare_data()

        if stage == "fit" or stage == "test":
            (
                self._train_dataset,
                self._val_dataset,
                self._test_dataset,
            ) = torch.utils.data.random_split(
                self._dataset,
                lengths=self._lengths,
                generator=self._generator,
            )

        if stage == "predict":
            self._predict_dataset = self._dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self._predict_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
