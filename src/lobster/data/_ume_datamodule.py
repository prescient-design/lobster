from pathlib import Path
from typing import Any, Callable, Dict, Sequence, TypeVar

import torch.utils.data
from beignet.datasets import ChEMBLDataset
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from lobster.constants import Modality
from lobster.datasets import (
    CalmDataset,
    FASTADataset,
    M320MDataset,
)
from lobster.tokenization import AminoAcidTokenizerFast, NucleotideTokenizerFast, SmilesTokenizerFast
from lobster.transforms import TokenizerTransform

from ._weighted_concat_sampler import WeightedConcatSampler

T = TypeVar("T")


SUPPORTED_DATASETS_MODALITIES_DICT = {
    "M320M": Modality.SMILES,
    "ChEMBL": Modality.SMILES,
    "Calm": Modality.NUCLEOTIDE,
    "Uniref50": Modality.AMINO_ACID,
}


class UmeLightningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        root: Path | str,
        tokenizer_max_length: int,
        datasets: None | Sequence[str] | Dict[str, float] = None,
        download: bool = False,
        use_text_descriptions: bool = False,
        generator: Generator | None = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        num_workers: int = 0,
        collate_fn: Callable[[list[T]], Any] | None = None,
        max_length: int = 512,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
        length: Sequence[float] = (0.9, 0.05, 0.05),
    ) -> None:
        """Lightning DataModule for handling multiple sequence datasets.

        Parameters
        ----------
        root : Path | str
            Root directory for dataset storage
        tokenizer_max_length : int
            Maximum length for tokenization
        datasets : None | Sequence[str] | Dict[str, float], optional
            Datasets to use. Can be:
            - None to use all supported datasets
            - List of dataset names for equal weighting
            - Dict mapping dataset names to sampling weights
        download : bool, default=False
            Whether to download datasets if not present
        use_text_descriptions : bool, default=False
            Whether to include text descriptions in dataset
        generator : Generator | None, default=None
            Random number generator for reproducibility
        seed : int, default=0xDEADBEEF
            Random seed if generator not provided
        batch_size : int, default=1
            Batch size for dataloaders
        num_workers : int, default=0
            Number of workers for data loading
        collate_fn : Callable[[list[T]], Any] | None, default=None
            Custom collate function for batching
        max_length : int, default=512
            Maximum sequence length
        pin_memory : bool, default=True
            Whether to pin memory for GPU transfer
        drop_last : bool, default=False
            Whether to drop last incomplete batch
        shuffle : bool, default=True
            Whether to shuffle training data
        length : Sequence[float], default=(0.9, 0.05, 0.05)
            Split ratios for train/val/test

        Examples
        --------
        Initialize with equal weights:
        >>> datamodule = UmeLightningDataModule(
        ...     root="/path/to/data",
        ...     tokenizer_max_length=512,
        ...     datasets=["M320M", "ChEMBL"]
        ... )

        Initialize with custom weights:
        >>> datamodule = UmeLightningDataModule(
        ...     root="/path/to/data",
        ...     tokenizer_max_length=512,
        ...     datasets={
        ...         "M320M": 0.5,
        ...         "ChEMBL": 2.0
        ...     }
        ... )
        """
        super().__init__()

        supported_datasets = set(SUPPORTED_DATASETS_MODALITIES_DICT.keys())

        if isinstance(datasets, dict):
            self.dataset_names = list(datasets.keys())
            self.weights = list(datasets.values())
        else:
            self.dataset_names = list(datasets) if datasets is not None else list(supported_datasets)
            self.weights = None

        if not set(self.dataset_names).issubset(supported_datasets):
            raise ValueError(
                f"Only the following datasets are supported: {supported_datasets}. "
                f"Unknown datasets: {set(self.dataset_names) - supported_datasets}"
            )

        self._root = root
        self._download = download
        self._tokenizer_max_length = tokenizer_max_length
        self._use_text_descriptions = use_text_descriptions
        self._generator = generator or Generator().manual_seed(seed)
        self._seed = seed
        self._batch_size = batch_size
        self._max_length = max_length
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._lengths = length

        # Initialize tokenizer transforms for each modality
        self._tokenizer_transforms = {
            Modality.SMILES: TokenizerTransform(
                SmilesTokenizerFast(), padding="max_length", truncation=True, max_length=self._tokenizer_max_length
            ),
            Modality.AMINO_ACID: TokenizerTransform(
                AminoAcidTokenizerFast(), padding="max_length", truncation=True, max_length=self._tokenizer_max_length
            ),
            Modality.NUCLEOTIDE: TokenizerTransform(
                NucleotideTokenizerFast(), padding="max_length", truncation=True, max_length=self._tokenizer_max_length
            ),
        }

        self.datasets = None
        self._train_datasets = None
        self._val_datasets = None
        self._test_datasets = None

    def _get_dataset(self, name: str) -> Dataset:
        """Get a dataset instance with appropriate tokenizer transform."""
        modality = SUPPORTED_DATASETS_MODALITIES_DICT[name]
        transform = self._tokenizer_transforms[modality]

        match name:
            case "M320M":
                return M320MDataset(
                    root=self._root,
                    download=self._download,
                    transform=transform,
                    columns=["smiles", "Description"] if self._use_text_descriptions else ["smiles"],
                )
            case "ChEMBL":
                return ChEMBLDataset(root=self._root, download=self._download, transform=transform)
            case "Calm":
                return CalmDataset(
                    root=self._root,
                    transform=transform,
                    columns=["sequence", "description"] if self._use_text_descriptions else ["sequence"],
                )
            case "Uniref50":
                return FASTADataset(
                    root=Path(self._root) / "uniref50.fasta",
                    transform=transform,
                    use_text_descriptions=self._use_text_descriptions,
                )
            case _:
                raise ValueError(f"Dataset {name} is not supported")

    def prepare_data(self) -> None:
        self.datasets = []
        for name in self.dataset_names:
            dataset = self._get_dataset(name)
            self.datasets.append(dataset)

    def setup(self, stage: str = "fit") -> None:
        if self.datasets is None:
            self.prepare_data()

        self._train_datasets = []
        self._val_datasets = []
        self._test_datasets = []

        for dataset in self.datasets:
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset,
                lengths=self._lengths,
                generator=self._generator,
            )

            self._train_datasets.append(train_dataset)
            self._val_datasets.append(val_dataset)
            self._test_datasets.append(test_dataset)

    def train_dataloader(self) -> DataLoader:
        sampler = WeightedConcatSampler(
            dataset_sizes=[len(d) for d in self._train_datasets], weights=self.weights, generator=self._generator
        )

        return DataLoader(
            ConcatDataset(self._train_datasets),
            batch_size=self._batch_size,
            sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ConcatDataset(self._val_datasets),
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            ConcatDataset(self._test_datasets),
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
