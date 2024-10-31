# Cast as dataset
from typing import Callable, Optional, Union

import pandas as pd
from lobster.transforms import Transform
from torch.utils.data import Dataset

SOURCE = ""


class AbAgSequencePPIDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame = None,
        source: str = None,
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
        sequence1_cols: tuple = ("fv_heavy", "fv_light"),
        sequence2_cols: tuple = ("antigen_sequence",),
        label_col: Optional[str] = None,
    ) -> None:
        if data is not None:
            self._data = data
        elif source is not None:
            print(f"Reading in data from source: {source}")
            self._data = pd.read_csv(source)
        elif source is None:
            print(f"Reading in data from default source: {SOURCE}")
            self._data = pd.read_csv(SOURCE)

        # allow for multiple columns to be combined into a single sequence
        if len(sequence1_cols) == 1:
            self._data["sequence_1"] = self._data[sequence1_cols[0]]
        elif len(sequence1_cols) == 2:
            self._data["sequence_1"] = (
                self._data[sequence1_cols[0]] + "-" + self._data[sequence1_cols[1]]
            )

        if len(sequence2_cols) == 1:
            self._data["sequence_2"] = self._data[sequence2_cols[0]]
        elif len(sequence2_cols) == 2:
            self._data["sequence_2"] = (
                self._data[sequence2_cols[0]] + "-" + self._data[sequence2_cols[1]]
            )

        if label_col is not None:
            self._data["label"] = self._data[label_col]
        else:
            self._data["label"] = self._data[
                "is_binder_pred"
            ]  # Nots used in eval, just a dummy label
        self._data = self._data[["sequence_1", "sequence_2", "label"]]

        self._transform_fn = transform_fn
        self._target_transform_fn = target_transform_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int):
        item = self._data.iloc[index]

        features = (item["sequence_1"], item["sequence_2"])

        if self._transform_fn is not None:
            features = self._transform_fn(features)

        target = item["label"]

        if self._target_transform_fn is not None:
            target = self._target_transform_fn(target)

        return features, target
