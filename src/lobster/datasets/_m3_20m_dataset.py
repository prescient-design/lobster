from pathlib import Path
from typing import Callable, Tuple

import pandas
import pooch
from beignet.transforms import Transform
from torch.utils.data import Dataset


class M320MDataset(Dataset):
    """Multi-Modal Molecular (M^3) Dataset with 20M compounds.
    M3-20M is a large-scale multi-modal molecular dataset with over 20 million molecules,
    integrating SMILES, molecular graphs, 3D structures, physicochemical properties,
    and textual descriptions. Reference: https://arxiv.org/abs/2412.06847

    This dataset version contains SMILES strings and text descriptions.

    Example:

    ```python
    dataset = M320MDataset()
    next(iter(dataset))

    (
        'CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1',
        'This molecule is a sulfoxide. It is functionally related to an albendazole...'
    )
    ```
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        download: bool = True,
        known_hash: str | None = None,
        transform: Callable | Transform | None = None,
        use_text_descriptions: bool = True,
    ):
        super().__init__()
        url = "https://huggingface.co/datasets/karina-zadorozhny/M320M-multi-modal-molecular-dataset/resolve/main/M320M-Dataset.parquet.gzip"

        suffix = ".parquet.gzip"

        if root is None:
            root = pooch.os_cache("lbster")

        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()

        if download:
            pooch.retrieve(
                url=url,
                fname=f"{self.__class__.__name__}{suffix}",
                known_hash=known_hash,
                path=root / self.__class__.__name__,
                progressbar=True,
            )

        self.data = pandas.read_parquet(root / self.__class__.__name__ / f"{self.__class__.__name__}{suffix}")

        self.columns = ["smiles"]

        if use_text_descriptions:
            self.columns += ["Description"]

        self.transform = transform

        self._x = self.data[self.columns].apply(tuple, axis=1)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        x = self._x[index]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self._x)
