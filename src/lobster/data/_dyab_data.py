import random
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

import torch
from lightning import LightningDataModule
from pandas import DataFrame
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Sampler

from ._dataframe_dataset_in_memory import (
    DataFrameDatasetInMemory,
)

T = TypeVar("T")


class DyAbDataFrameLightningDataModule(LightningDataModule):
    def __init__(
        self,
        data: DataFrame = None,
        *,
        remove_nulls: bool | None = False,
        transform_fn: Callable | None = None,
        target_transform_fn: Callable | None = None,
        joint_transform_fn: Callable | None = None,
        lengths: Sequence[float] | None = (0.9, 0.05, 0.05),
        generator: Generator | None = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Iterable | Sampler | None = None,
        batch_sampler: Iterable[Sequence] | Sampler[Sequence] | None = None,
        collate_fn: Callable[[list[T]], Any] | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        max_length: int = 512,
        mlm: bool = True,
        lead_vh: str | None = None,
        lead_vl: str | None = None,
        lead_pkd: float | None = None,
    ) -> None:
        """
        :param transform_fn: A ``Callable`` that maps a sequence to a
            transformed sequence (default: ``None``).

        :param target_transform_fn: ``Callable`` that maps a target (a cluster
            identifier) to a transformed target (default: ``None``).

        :param joint_transform_fn: ``Callable`` that maps a feature and target
            to a transformed feature, target (default: ``None``). Needed in cases
            where the target transform is dependent on the features or vice versa

        :param lengths: Fractions of splits to generate.

        :param generator: Generator used for the random permutation (default:
            ``None``).

        :param seed: Desired seed. Value must be within the inclusive range
            ``[-0x8000000000000000, 0xFFFFFFFFFFFFFFFF]`` (default:
            ``0xDEADBEEF``). Otherwise, a ``RuntimeError`` is raised. Negative
            inputs are remapped to positive values with the formula
            ``0xFFFFFFFFFFFFFFFF + seed``.

        :param batch_size: Samples per batch (default: ``1``).

        :param shuffle: If ``True``, reshuffle datasets at every epoch (default:
            ``True``).

        :param sampler: Strategy to draw samples from the dataset (default:
            ``None``). Can be any ``Iterable`` with ``__len__`` implemented.
            If specified, :attr:`shuffle` must be ``False``.

        :param batch_sampler: :attr:`sampler`, but returns a batch of indices
            (default: ``None``). Mutually exclusive with :attr:`batch_size`,
            :attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.

        :param num_workers: Subprocesses to use (default: ``0``). ``0`` means
            that the datasets will be loaded in the main process.

        :param collate_fn: Merges samples to form a mini-batch of Tensor(s)
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

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._data = data
        if transform_fn is not None:
            self._transform_fn = transform_fn
        self._target_transform_fn = target_transform_fn
        self._joint_transform_fn = joint_transform_fn
        self._lengths = lengths
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._sampler = sampler
        self._batch_sampler = batch_sampler
        self._collate_fn = collate_fn
        self._num_workers = num_workers
        self._max_length = max_length
        self._mlm = mlm

        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._remove_nulls = remove_nulls
        self._lead_vh = lead_vh
        self._lead_vl = lead_vl
        self._lead_pkd = lead_pkd

        self._dataset = None

    def prepare_data(self) -> None:
        # Load in Dataset, transform sequences
        dataset = DyAbDataFrameDatasetInMemory(
            data=self._data,
            columns=["fv_heavy", "fv_light"],
            target_columns=["pKD"],
        )

        self._dataset = dataset

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        if self._dataset is None:
            self.prepare_data()

        if stage == "fit":
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

        if stage == "rank":
            if self._lead_pkd is None or self._lead_vh is None or self._lead_vl is None:
                raise ValueError("cannot rank without a lead design to rank against")
            self._rank_dataset = DyAbDataFrameDatasetInMemory(
                data=self._data,
                columns=["fv_heavy", "fv_light"],
                target_columns=["pKD"],
                ranking=True,
                lead_vh=self._lead_vh,
                lead_vl=self._lead_vl,
                lead_pkd=self._lead_pkd,
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

    def rank_dataloader(self) -> DataLoader:
        return DataLoader(
            self._rank_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )


class DyAbDataFrameDatasetInMemory(DataFrameDatasetInMemory):
    def __init__(
        self,
        data: DataFrame,
        columns: Sequence[str] | None = None,
        target_columns: Sequence[str] | None = None,
        ranking: bool = False,
        lead_vh: str | None = None,
        lead_vl: str | None = None,
        lead_pkd: float | None = None,
    ):
        super().__init__(data, columns=columns, target_columns=target_columns)
        self._data = data
        self._columns = columns if columns is not None else None
        self._target_columns = target_columns if target_columns is not None else None
        self._can_rank = False
        if ranking:
            if not lead_vh or not lead_vl or not lead_pkd:
                raise ValueError("cannot rank without all lead design information")
            self._lead = DataFrame(
                {
                    "fv_heavy": [lead_vh],
                    "fv_light": [lead_vl],
                    "pKD": [lead_pkd],
                }
            )
            self._can_rank = True

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        item1 = self._data.iloc[index]
        item2 = self._data.sample(n=1).iloc[0]
        if self._can_rank:
            item2 = self._lead.iloc[0]

        if len(self._columns) > 1:
            x1 = tuple(item1[col] for col in self._columns)
            x2 = tuple(item2[col] for col in self._columns)
        else:
            x1 = item1[self._columns[0]]
            x2 = item2[self._columns[0]]

        # Apply transform
        if self._transform_fn is not None:
            x1 = self._transform_fn(x1)
            x2 = self._transform_fn(x2)

        if self._target_columns is None:
            return x1, x2

        # Apply target transform if target columns are present
        if len(self._target_columns) > 1:
            y1 = tuple(item1[col] for col in self._target_columns)
            y2 = tuple(item2[col] for col in self._target_columns)
        else:
            y1 = item1[self._target_columns[0]]
            y2 = item2[self._target_columns[0]]

        if self._target_transform_fn is not None:
            y1 = self._target_transform_fn(y1)
            y2 = self._target_transform_fn(y2)

        if len(self._target_columns) > 1 and not all(isinstance(y_val, Tensor) for y_val in y1):
            y1 = tuple(Tensor(y_val) for y_val in y1)
            y2 = tuple(Tensor(y_val) for y_val in y2)

        elif not isinstance(y1, Tensor):
            y1 = torch.as_tensor(y1)
            y2 = torch.as_tensor(y2)

        return x1, x2, y1, y2
