from pathlib import Path
from typing import Callable, ClassVar, Sequence, Tuple

import pandas as pd
import pooch
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset

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
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            data = load_dataset("ncfrey/calm", split="train")
            data.to_parquet(dataset_path)
            self.data = data
        else:
            self.data = pd.read_parquet(dataset_path)

        self.columns = ["sequence", "description"] if columns is None else columns
        self.transform = transform

        self._x = list(self.data[self.columns].apply(tuple, axis=1))

    def __getitem__(self, index: int) -> Tuple[str, str]:
        x = self._x[index]

        if len(x) == 1:
            x = x[0]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self) -> int:
        return len(self._x)


class CalmIterableDataset(IterableDataset):
    """
    Iterable Dataset from Outeiral, C., Deane, C.M.
    Codon language embeddings provide strong signals for use in protein engineering.
    Nat Mach Intell 6, 170–179 (2024). https://doi.org/10.1038/s42256-024-00791-0

    Training data is from FASTA files containing coding DNA sequences.

    Example:

    ```python
    dataset = CalmIterableDataset()
    next(iter(dataset))

    (
        'ATGCCTCCAAAGTACTTTGAAGTAGAAAAAGAATTCAAAA...',
        'Homo sapiens solute carrier family 28 member 3 (SLC28A3)'
    )
    ```
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        keys: Sequence[str] | None = None,
        split: str = "train",
    ):
        """
        Initialize the CalmIterableDataset.

        Parameters
        ----------
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample.
        keys : Sequence of str or None, optional
            List of keys to be used from the dataset.
        split : str, optional
            Which split of the dataset to use (only 'train' is available).
        """
        super().__init__()

        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(f"Split '{split}' is not supported.")

        self.dataset = load_dataset(
            "ncfrey/calm",
            split="train",
            cache_dir=root,
        )
        self.dataset = self.dataset.to_iterable_dataset()

        self.keys = ["sequence", "description"] if keys is None else keys
        self.transform = transform

    def __iter__(self):
        for sample in self.dataset:
            x = tuple(sample[key] for key in self.keys)

            if len(x) == 1:
                x = x[0]

            if self.transform is not None:
                x = self.transform(x)

            yield x
