from pathlib import Path
from typing import Callable, Sequence, Tuple

import pandas as pd
import pooch
from datasets import load_dataset
from torch.utils.data import Dataset

from lobster.transforms import Transform


class CalmDataset(Dataset):
    """
    Dataset from Outeiral, C., Deane, C.M.
    Codon language embeddings provide strong signals for use in protein engineering.
    Nat Mach Intell 6, 170–179 (2024). https://doi.org/10.1038/s42256-024-00791-0

    Training data is from FASTA files containing coding DNA sequences.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        download: bool = True,
        known_hash: str | None = None,
        transform: Callable | Transform | None = None,
        columns: Sequence[str] | None = None,
    ):
        super().__init__()

        if root is None:
            root = pooch.os_cache("lbster")

        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        dataset_path = root / self.__class__.__name__ / f"{self.__class__.__name__}"
        if not dataset_path.exists():
            data = load_dataset("ncfrey/calm", split="train")
            data.to_parquet(dataset_path)
            self.data = data
        else:
            self.data = pd.read_parquet(dataset_path)

        self.columns = ["sequence", "description"] if columns is None else columns
        self.transform = transform

        self._x = self.data[self.columns].apply(tuple, axis=1)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        x = self._x[index]

        if len(x) == 1:
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self._x)
