from collections.abc import Callable
from pathlib import Path

import pandas
import pooch
import torch
from beignet.transforms import Transform
from torch import Tensor
from torch.utils.data import Dataset

from lobster.constants import MOLECULEACE_TASKS


class MoleculeACEDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        task: str,
        download: bool = True,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
        train: bool = True,
        known_hash: str | None = None,
    ) -> None:
        """
        Molecule Activity Cliff Estimation (MoleculeACE) from Tilborg et al. (2024)

        Reference: https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073

        Contains activity data for 30 different ChEMBL targets (=tasks).
        """
        super().__init__()

        if root is None:
            root = pooch.os_cache("lbster")

        if isinstance(root, str):
            root = Path(root)

        self._root = root.resolve()

        self._download = download
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn
        self.column = "smiles"
        self.target_column = "y [pEC50/pKi]"
        self.task = task
        self.train = train

        if self.task not in MOLECULEACE_TASKS:
            raise ValueError(f"`task` must be one of {MOLECULEACE_TASKS}, got {self.task}")

        suffix = ".csv"
        url = "https://raw.githubusercontent.com/molML/MoleculeACE/main/MoleculeACE/Data/benchmark_data/"
        url = f"{url}/{self.task}{suffix}"

        if self._download:
            pooch.retrieve(
                url=url,
                fname=f"{self.__class__.__name__}_{self.task}_{suffix}",
                known_hash=known_hash,
                path=self._root / self.__class__.__name__,
                progressbar=True,
            )

        data = pandas.read_csv(
            self._root / self.__class__.__name__ / f"{self.__class__.__name__}_{self.task}_{suffix}"
        ).reset_index(drop=True)

        if train:
            self.data = data[data["split"] == "train"]
        else:
            self.data = data[data["split"] == "test"]

    def __getitem__(self, index: int) -> tuple[str | Tensor, Tensor]:
        item = self.data.iloc[index]

        x = item[self.column]

        if self.transform_fn is not None:
            x = self.transform_fn(x)

        y = item[self.target_column]

        if self.target_transform_fn is not None:
            y = self.target_transform_fn(y)

        if not isinstance(y, Tensor):
            y = torch.tensor(y).unsqueeze(-1)

        return x, y

    def __len__(self) -> int:
        return len(self.data)
