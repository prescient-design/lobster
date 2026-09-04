from collections.abc import Callable
from pathlib import Path

import pandas
import pooch
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lobster.constants import MOLECULENET_TASK_NAMES, MOLECULENET_TASKS
from lobster.transforms import Transform

MOLECULENET_BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"


class MoleculeNetDataset(Dataset):
    """MoleculeNet single-task datasets from Wu et al. (2018).

    Loads SMILES and target columns from DeepChem-hosted CSV files. Datasets do
    not expose predefined train/test splits; callers (e.g.
    :class:`~lobster.callbacks.MoleculeNetSklearnProbeCallback`) should perform
    random splitting.

    Reference:
        Wu et al. (2018) "MoleculeNet: a benchmark for molecular machine learning"
        https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a

    Data source:
        https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        task: str,
        download: bool = True,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
        known_hash: str | None = None,
    ) -> None:
        super().__init__()

        if root is None:
            root = pooch.os_cache("lbster")

        if isinstance(root, str):
            root = Path(root)

        self._root = root.resolve()
        self._download = download
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn
        self.task = task

        if self.task not in MOLECULENET_TASK_NAMES:
            raise ValueError(f"`task` must be one of {sorted(MOLECULENET_TASK_NAMES)}, got {self.task}")

        task_type, _num_classes, filename, smiles_column, target_column = MOLECULENET_TASKS[self.task]
        self.task_type = task_type
        self.column = smiles_column
        self.target_column = target_column

        url = f"{MOLECULENET_BASE_URL}{filename}"
        cache_fname = f"{self.__class__.__name__}_{filename}"

        if self._download:
            pooch.retrieve(
                url=url,
                fname=cache_fname,
                known_hash=known_hash,
                path=self._root / self.__class__.__name__,
                progressbar=True,
            )

        data = pandas.read_csv(self._root / self.__class__.__name__ / cache_fname).reset_index(drop=True)

        # Keep only SMILES + target; drop rows with missing values
        data = data[[self.column, self.target_column]].copy()
        data[self.target_column] = pandas.to_numeric(data[self.target_column], errors="coerce")
        data.dropna(subset=[self.column, self.target_column], inplace=True)
        data.reset_index(drop=True, inplace=True)

        if self.task_type == "binary":
            data[self.target_column] = data[self.target_column].astype(int)
        else:
            data[self.target_column] = data[self.target_column].astype(float)

        self.data = data

    def __getitem__(self, index: int) -> tuple[str | Tensor, Tensor]:
        item = self.data.iloc[index]

        x = item[self.column]
        if self.transform_fn is not None:
            x = self.transform_fn(x)

        y = item[self.target_column]
        if self.target_transform_fn is not None:
            y = self.target_transform_fn(y)

        if not isinstance(y, Tensor):
            if self.task_type == "binary":
                y = torch.tensor(y, dtype=torch.long).unsqueeze(-1)
            else:
                y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        return x, y

    def __len__(self) -> int:
        return len(self.data)
