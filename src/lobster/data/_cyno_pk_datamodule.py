from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar, Union

import torch.utils.data
from lightning import LightningDataModule

from ._imports import _PRESCIENT_AVAILABLE

if _PRESCIENT_AVAILABLE:
    from prescient.datasets import CynoPKClearanceDataset


from beignet.transforms import Transform
from torch import Generator
from torch.utils.data import DataLoader, Sampler

T = TypeVar("T")


class CynoPKClearanceLightningDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        download: bool = False,
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
        lengths: Union[Sequence[float], None] = (0.8, 0.1, 0.1),
        generator: Union[Generator, None] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Union[Iterable, Sampler, None] = None,
        batch_sampler: Union[Iterable[Sequence], Sampler[Sequence], None] = None,
        num_workers: int = 0,
        collate_fn: Union[Callable[[T], Any], None] = None,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param download: If ``True``, download the dataset to the :attr:`root`
            directory (default: ``False``). If the dataset is already
            downloaded, it is not redownloaded.

        :param transform_fn: A ``Callable`` or ``Transform`` that maps data to
            transformed data (default: ``None``).

        :param target_transform_fn: ``Callable`` or ``Transform`` that maps a
            target to a transformed target (default: ``None``).

        :param lengths: Fractions of splits to generate.

        :param generator: Generator used for the random permutation (default:
            ``None``).

        :param seed: Desired seed. Value must be within the inclusive range
            ``[-0x8000000000000000, 0xFFFFFFFFFFFFFFFF]`` (default:
            ``0xDEADBEEF``). Otherwise, a ``RuntimeError`` is raised. Negative
            inputs are remapped to positive values with the formula
            ``0xFFFFFFFFFFFFFFFF + seed``.

        :param batch_size: Samples per batch (default: ``1``).

        :param shuffle: If ``True``, reshuffle datasets at every epoch
            (default: ``True``).

        :param sampler: Strategy to draw samples from the dataset (default:
            ``None``). Can be any ``Iterable`` with ``__len__`` implemented.
            If specified, :attr:`shuffle` must be ``False``.

        :param batch_sampler: :attr:`sampler`, but returns a batch of indices
            (default: ``None``). Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.

        :param num_workers: Subprocesses to use (default: ``0``). ``0`` means
            that the datasets will be loaded in the main process.

        :param collate_fn: Merges samples to form a mini-batch of ``Tensor``s
            (default: ``None``).

        :param pin_memory: If ``True``, Tensors are copied to the device's
            (e.g., CUDA) pinned memory before returning them (default:
            ``True``).

        :param drop_last: If ``True``, drop the last incomplete batch, if the
            dataset size is not divisible by the batch size (default:
            ``False``). If ``False`` and the size of dataset is not divisible
            by the batch size, then the last batch will be smaller.
        """
        super().__init__()

        if lengths is None:
            lengths = (0.8, 0.1, 0.1)

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._root = root
        self._download = download
        self._transform_fn = transform_fn
        self._target_transform_fn = target_transform_fn
        self._lengths = lengths
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._sampler = sampler
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last

    def prepare_data(self) -> None:
        _dataset = CynoPKClearanceDataset(  # download data
            root=self._root,
            download=self._download,
            columns=["fv_heavy", "fv_light"],
            target_columns=["cl_mean"],
            transform_fn=self._transform_fn,
            target_transform_fn=self._target_transform_fn,
        )

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        dataset = CynoPKClearanceDataset(
            root=self._root,
            download=False,  # prepare_data already downloads the dataset
            columns=["fv_heavy", "fv_light"],
            target_columns=["cl_mean"],
            transform_fn=self._transform_fn,
            target_transform_fn=self._target_transform_fn,
        )
        if stage == "fit" or stage is None:
            (
                self._train_dataset,
                self._val_dataset,
                self._test_dataset,
            ) = torch.utils.data.random_split(
                dataset,
                lengths=self._lengths,
                generator=self._generator,
            )

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
            shuffle=self._shuffle,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
