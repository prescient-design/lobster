import random
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar, Union

import torch
from lightning import LightningDataModule
from pandas import DataFrame
from torch import Generator, Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from lobster.transforms import Transform

T = TypeVar("T")


class DataFrameLightningDataModule(LightningDataModule):
    def __init__(
        self,
        data: DataFrame = None,
        *,
        remove_nulls: Optional[bool] = False,
        transform_fn: Optional[Callable] = None,
        target_transform_fn: Optional[Callable] = None,
        joint_transform_fn: Optional[Callable] = None,
        lengths: Optional[Sequence[float]] = (0.9, 0.05, 0.05),
        generator: Optional[Generator] = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Optional[Union[Iterable, Sampler]] = None,
        batch_sampler: Optional[Union[Iterable[Sequence], Sampler[Sequence]]] = None,
        collate_fn: Optional[Callable[[list[T]], Any]] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        max_length: int = 512,
        mlm: bool = True,
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

        self._dataset = None

    def prepare_data(self) -> None:
        # Load in Dataset, transform sequences
        dataset = DataFrameDatasetInMemory(
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


class DataFrameDatasetInMemory(Dataset):
    _data: DataFrame

    def __init__(
        self,
        data: DataFrame,
        *,
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
        columns: Optional[Sequence[str]] = None,
        target_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        :param root: Root directory where the dataset subdirectory exists or,
            if :attr:`download` is ``True``, the directory where the dataset
            subdirectory will be created and the dataset downloaded.

        :param transform_fn: A ``Callable`` or ``Transform`` that maps data to
            transformed data (default: ``None``).

        :param target_transform_fn: ``Callable`` or ``Transform`` that maps a
            target to a transformed target (default: ``None``).
        """
        self._transform_fn = transform_fn

        self._target_transform_fn = target_transform_fn

        self._data = data

        self._columns = columns if columns is not None else None
        self._target_columns = target_columns if target_columns is not None else None

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> T:
        item = self._data.iloc[index]

        if len(self._columns) > 1:
            x = tuple(item[col] for col in self._columns)
        else:
            x = item[self._columns[0]]

        # Apply transform
        if self._transform_fn is not None:
            x = self._transform_fn(x)

        if self._target_columns is None:
            return x

        # Apply target transform if target columns are present
        if len(self._target_columns) > 1:
            y = tuple(item[col] for col in self._target_columns)
        else:
            y = item[self._target_columns[0]]

        if self._target_transform_fn is not None:
            y = self._target_transform_fn(y)

        if len(self._target_columns) > 1 and not all(isinstance(y_val, Tensor) for y_val in y):
            y = tuple(Tensor(y_val) for y_val in y)

        elif not isinstance(y, Tensor):
            y = torch.as_tensor(y)

        return x, y
