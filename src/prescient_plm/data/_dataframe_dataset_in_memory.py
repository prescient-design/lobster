from typing import Callable, Optional, Sequence, TypeVar, Union

from pandas import DataFrame
from prescient.transforms import Transform
from torch import Tensor
from torch.utils.data import Dataset

T = TypeVar("T")


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
            y = Tensor(y)

        return x, y
