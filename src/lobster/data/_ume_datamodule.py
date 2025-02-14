from pathlib import Path
from typing import Any, Callable, Dict, Literal, Sequence, TypeVar

import torch.utils.data
from beignet.datasets import ChEMBLDataset
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import ConcatDataset, DataLoader

from lobster.datasets import (
    CalmDataset,
    FASTADataset,
    M320MDataset,
)
from lobster.tokenization import AminoAcidTokenizerFast, NucleotideTokenizerFast, SmilesTokenizerFast
from lobster.transforms import TokenizerTransform

from ._weighted_concat_sampler import WeightedConcatSampler

T = TypeVar("T")

SUPPORTED_DATASETS = {"M320M", "ChEMBL", "Calm", "Uniref50"}


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
        super().__init__()

        if isinstance(datasets, dict):
            self.dataset_names = list(datasets.keys())
            self.weights = list(datasets.values())
        else:
            self.dataset_names = list(datasets) if datasets is not None else list(SUPPORTED_DATASETS)
            self.weights = [1.0] * len(self.dataset_names)

        if not set(self.dataset_names).issubset(SUPPORTED_DATASETS):
            raise ValueError(
                f"Only the following datasets are supported: {SUPPORTED_DATASETS}. "
                f"Unknown datasets: {set(self.dataset_names) - SUPPORTED_DATASETS}"
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

        self._datasets = None
        self._train_datasets = None
        self._val_datasets = None
        self._test_datasets = None

    def _get_tokenizer_transform(self, modality: Literal["SMILES", "amino_acid", "nucleotide"]) -> None:
        match modality:
            case "SMILES":
                tokenizer = SmilesTokenizerFast()
            case "amino_acid":
                tokenizer = AminoAcidTokenizerFast()
            case "nucleotide":
                tokenizer = NucleotideTokenizerFast()

        return TokenizerTransform(
            tokenizer, padding="max_length", truncation=True, max_length=self._tokenizer_max_length
        )

    def prepare_data(self) -> None:
        smiles_transform = self._get_tokenizer_transform("SMILES")
        amino_acid_transform = self._get_tokenizer_transform("amino_acid")
        nucleotide_transform = self._get_tokenizer_transform("nucleotide")

        self._datasets = []

        # SMILES
        if "M320M" in self.dataset_names:
            self._datasets.append(
                M320MDataset(
                    root=self._root,
                    download=self._download,
                    transform=smiles_transform,
                    columns=["smiles", "Description"] if self._use_text_descriptions else ["smiles"],
                )
            )

        # SMILES
        if "ChEMBL" in self.dataset_names:
            self._datasets.append(ChEMBLDataset(root=self._root, download=self._download, transform=smiles_transform))

        # Nucleotide
        if "Calm" in self.dataset_names:
            self._datasets.append(
                CalmDataset(
                    root=self._root,
                    transform=nucleotide_transform,
                    columns=["sequence", "description"] if self._use_text_descriptions else ["sequence"],
                )
            )

        # Amino acid
        if "Uniref50" in self.dataset_names:
            self._datasets.append(
                FASTADataset(
                    root=Path(self._root) / "uniref50.fasta",
                    transform=amino_acid_transform,
                    use_text_descriptions=self._use_text_descriptions,
                )
            )

    def setup(self, stage: str = "fit") -> None:
        if self._datasets is None:
            self.prepare_data()

        # Split each dataset individually
        self._train_datasets = []
        self._val_datasets = []
        self._test_datasets = []

        for dataset in self._datasets:
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset,
                lengths=self._lengths,
                generator=self._generator,
            )

            self._train_datasets.append(train_dataset)
            self._val_datasets.append(val_dataset)
            self._test_datasets.append(test_dataset)

    def train_dataloader(self) -> DataLoader:
        # Sampler to weight datasets according to provided weights
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

    def val_dataloader(self):
        return DataLoader(
            ConcatDataset(self._val_datasets),
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            ConcatDataset(self._test_datasets),
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
