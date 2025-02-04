import random
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import torch.utils.data
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.datasets._calm_dataset import CalmDataset
from lobster.tokenization import NucleotideTokenizerFast
from lobster.transforms import TokenizerTransform, Transform

T = TypeVar("T")


class CalmLightningDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path] = None,
        *,
        use_text_descriptions: bool = False,
        download: bool = False,
        transform_fn: Union[Callable, Transform, None] = None,
        lengths: Optional[Sequence[float]] = (0.9, 0.05, 0.05),
        generator: Optional[Generator] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Optional[Union[Iterable, Sampler]] = None,
        batch_sampler: Optional[Union[Iterable[Sequence], Sampler[Sequence]]] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[list[T]], Any]] = None,
        max_length: int = 512,
        pin_memory: bool = True,
        drop_last: bool = False,
        train: bool = True,
    ) -> None:
        """
        Calm Lightning DataModule.

        Parameters
        ----------
        root : Union[str, Path], optional
            Root directory of the dataset.
        use_text_descriptions : bool, optional
            Whether to use text descriptions, by default False.
        download : bool, optional
            Whether to download the dataset, by default False.
        transform_fn : Union[Callable, Transform, None], optional
            Function or transform to apply to the data, by default None.
        lengths : Optional[Sequence[float]], optional
            Sequence of lengths for splitting the dataset, by default (0.9, 0.05, 0.05).
        generator : Optional[Generator], optional
            Random generator for shuffling, by default None.
        seed : int, optional
            Seed for the random generator, by default 0xDEADBEEF.
        batch_size : int, optional
            Number of samples per batch, by default 1.
        shuffle : bool, optional
            Whether to shuffle the data, by default True.
        sampler : Optional[Union[Iterable, Sampler]], optional
            Sampler for data loading, by default None.
        batch_sampler : Optional[Union[Iterable[Sequence], Sampler[Sequence]]], optional
            Batch sampler for data loading, by default None.
        num_workers : int, optional
            Number of worker processes for data loading, by default 0.
        collate_fn : Optional[Callable[[list[T]], Any]], optional
            Function to merge a list of samples to form a mini-batch, by default None.
        max_length : int, optional
            Maximum length of sequences, by default 512.
        pin_memory : bool, optional
            Whether to pin memory during data loading, by default True.
        drop_last : bool, optional
            Whether to drop the last incomplete batch, by default False.
        train : bool, optional
            Whether the module is in training mode, by default True.


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
        self._use_text_descriptions = use_text_descriptions

        if transform_fn is None and not use_text_descriptions:
            transform_fn = TokenizerTransform(
                tokenizer=NucleotideTokenizerFast(),
                padding="max_length",
                max_length=512,
                truncation=True,
            )

        self._transform_fn = transform_fn

    def prepare_data(self) -> None:
        dataset = CalmDataset(
            root=self._root,
            download=self._download,
            transform=self._transform_fn,
            columns=["sequence", "description"] if self._use_text_descriptions else ["sequence"],
        )

        self._dataset = dataset

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
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
