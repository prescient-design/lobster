from pathlib import Path
from typing import Callable, Sequence, Tuple

import pandas
import pooch
from beignet.transforms import Transform
from torch.utils.data import Dataset


class M320MDataset(Dataset):
    """Multi-Modal Molecular (M^3) Dataset with 20M compounds.

    Reference: https://arxiv.org/abs/2412.06847

    Contains SMILES and text descriptions.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        known_hash: str | None = None,
        *,
        download: bool = False,
        full_dataset: bool = False,
        columns: Sequence[str] | None = None,
        transform: Callable | Transform | None = None,
        target_transform: Callable | Transform | None = None,
    ):
        super().__init__()
        url_base = "https://huggingface.co/datasets/Alex99Gsy/M-3_Multi-Modal-Molecule/resolve/main/"

        if full_dataset:
            url = url_base + "M%5E3_Multi.rar"

            if root is None:
                root = pooch.os_cache("lbster")

            if isinstance(root, str):
                root = Path(root)

            self.root = root.resolve()

            if download:
                pooch.retrieve(
                    url=url,
                    fname=f"{self.__class__.__name__}.rar",
                    known_hash=known_hash,
                    path=root / self.__class__.__name__,
                    progressbar=True,
                )
                # TODO

        else:
            url = url_base + "M%5E3_Multi.csv"
            self.data = pandas.read_csv(url)

        self.columns = ["smiles_x", "Description"] if columns is None else columns
        self.transform = transform
        self.target_transform = target_transform

        self._x = self.data[self.columns].apply(tuple, axis=1)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        x = self._x[index]

        if len(x) == 1:
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self._data)
