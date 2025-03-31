from pathlib import Path
from typing import Callable, ClassVar, Literal, Sequence, Tuple

import pandas as pd
import pooch
from datasets import load_dataset
from torch.utils.data import Dataset

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset
from lobster.transforms import Transform


class CalmDataset(Dataset):
    """
    Dataset from Outeiral, C., Deane, C.M.
    Codon language embeddings provide strong signals for use in protein engineering.
    Nat Mach Intell 6, 170–179 (2024). https://doi.org/10.1038/s42256-024-00791-0

    Training data is originally from FASTA files containing coding DNA sequences.
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "train_iid", "val_iid", "test_iid", "heldout"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        columns: Sequence[str] | None = None,
        split: Literal["train", "train_iid", "val_iid", "test_iid", "heldout"] = "train",
    ):
        """
        Parameters
        ----------
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample.
        columns : Sequence of str or None, optional
            List of columns to be used from the dataset. Default is ["sequence", "description"].
        split : str, optional
            Which split of the dataset to use. Options are:
            - "train": Original training data from paper
            - "train_iid": ~80% of training data (random split)
            - "val_iid": ~10% of training data (random split)
            - "test_iid": ~10% of training data (random split)
            - "heldout": Heldout test data, from paper
        """
        super().__init__()

        if root is None:
            root = pooch.os_cache("lbster")

        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()
        self.split = split

        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(f"Split '{split}' not supported. Choose from {self.SUPPORTED_SPLITS}.")

        dataset_path = root / self.__class__.__name__ / f"{self.__class__.__name__}_{split}"
        if not dataset_path.exists():
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            data = load_dataset("taylor-joren/calm", data_files=f"{split}/*.parquet", split="train")
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


class CalmIterableDataset(HuggingFaceIterableDataset):
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

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "train_iid", "val_iid", "test_iid", "heldout"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        keys: Sequence[str] | None = None,
        split: Literal["train", "train_iid", "val_iid", "test_iid", "heldout"] = "train",
        download: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 1000,
        limit: int | None = None,
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
            Which split of the dataset to use. Options are:
            - "train": Original training data
            - "train_iid": ~80% of training data (random split)
            - "val_iid": ~10% of training data (random split)
            - "test_iid": ~10% of training data (random split)
            - "heldout": Heldout test data
        limit : int or None, optional
            Limit the number of samples to load.
        """
        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(f"Split '{split}' not supported. Choose from {self.SUPPORTED_SPLITS}.")

        super().__init__(
            dataset_name="taylor-joren/calm",
            data_files=f"{split}/*.parquet",
            root=root,
            transform=transform,
            keys=keys or ["sequence", "description"],
            split="train",  # Since we're using data_files to specify the split
            shuffle=shuffle,
            download=download,
            shuffle_buffer_size=shuffle_buffer_size,
            limit=limit,
        )

    def _passes_type_check(self, sample: tuple[str]) -> bool:
        return all(isinstance(s, str) for s in sample)
