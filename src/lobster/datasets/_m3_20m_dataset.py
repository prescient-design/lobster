from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import pandas
import pooch
from beignet.transforms import Transform
from torch.utils.data import Dataset

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset


class M320MIterableDataset(HuggingFaceIterableDataset):
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

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "validation", "test"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        keys: Sequence[str] | None = None,
        split: str = "train",
        download: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 1000,
        limit: int | None = None,
    ):
        """
        Initialize the M320MDataset.

        Parameters
        ----------
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
        download : bool, optional
            If True, download the dataset if not already present.
        known_hash : str or None, optional
            Known hash of the dataset file for verification.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample.
        columns : list of str or None, optional
            List of columns to be used from the dataset.
        limit : int or None, optional
            Limit the number of samples to load.
        """
        super().__init__(
            dataset_name="karina-zadorozhny/M320M",
            root=root,
            transform=transform,
            keys=keys or ["smiles", "Description"],
            split=split,
            download=download,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            limit=limit,
        )

    def _passes_type_check(self, sample: tuple[str]) -> bool:
        return all(isinstance(s, str) for s in sample)


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
        columns: list[str] | None = None,
    ):
        """
        Initialize the M320MDataset.

        Parameters
        ----------
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
        download : bool, optional
            If True, download the dataset if not already present.
        known_hash : str or None, optional
            Known hash of the dataset file for verification.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample.
        columns : list of str or None, optional
            List of columns to be used from the dataset.
        """
        super().__init__()
        url = "https://huggingface.co/datasets/karina-zadorozhny/M320M/resolve/main/M320M-Dataset.parquet.gzip"

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

        self.columns = ["smiles", "Description"] if columns is None else columns

        self.transform = transform

        self._x = list(self.data[self.columns].apply(tuple, axis=1))

    def __getitem__(self, index: int) -> tuple[str, str]:
        x = self._x[index]

        if len(x) == 1:
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self._x)
