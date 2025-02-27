from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Type

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import ChainDataset, DataLoader, Dataset, IterableDataset

from lobster.constants import Modality, Split
from lobster.datasets import AMPLIFYIterableDataset, CalmIterableDataset, M320MIterableDataset
from lobster.tokenization import AminoAcidTokenizerFast, NucleotideTokenizerFast, SmilesTokenizerFast
from lobster.transforms import TokenizerTransform


@dataclass
class DatasetInfo:
    dataset_class: Type
    modality: Modality
    supported_splits: set[Split]


SUPPORTED_DATASETS_INFO = {
    "M320M": DatasetInfo(
        dataset_class=M320MIterableDataset,
        modality=Modality.SMILES,
        supported_splits={Split.TRAIN, Split.VALIDATION, Split.TEST},
    ),
    "Calm": DatasetInfo(
        dataset_class=CalmIterableDataset,
        modality=Modality.NUCLEOTIDE,
        supported_splits={Split.TRAIN},
    ),
    "AMPLIFY": DatasetInfo(
        dataset_class=AMPLIFYIterableDataset,
        modality=Modality.AMINO_ACID,
        supported_splits={Split.TRAIN, Split.TEST},
    ),
}


class UmeLightningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        tokenizer_max_length: int,
        datasets: None | Sequence[str] = None,
        root: Path | str | None = None,
        seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 10000,
    ) -> None:
        super().__init__()

        supported_datasets = set(SUPPORTED_DATASETS_INFO.keys())

        self.dataset_names: list[str] = list(datasets) if datasets is not None else list(supported_datasets)

        if not set(self.dataset_names).issubset(supported_datasets):
            raise ValueError(
                f"Only the following datasets are supported: {supported_datasets}. "
                f"Unknown datasets: {set(self.dataset_names) - supported_datasets}"
            )

        self._root = root
        self._tokenizer_max_length = tokenizer_max_length
        self._generator = Generator().manual_seed(seed)
        self._seed = seed
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._shuffle_buffer_size = shuffle_buffer_size

        # Initialize tokenizer transforms for each modality
        tokenizer_instances = {
            Modality.SMILES: SmilesTokenizerFast(),
            Modality.AMINO_ACID: AminoAcidTokenizerFast(),
            Modality.NUCLEOTIDE: NucleotideTokenizerFast(),
        }

        self._tokenizer_transforms = {
            modality: TokenizerTransform(
                tokenizer, padding="max_length", truncation=True, max_length=self._tokenizer_max_length
            )
            for modality, tokenizer in tokenizer_instances.items()
        }

        self._train_datasets: list[IterableDataset] = []
        self._val_datasets: list[IterableDataset] = []
        self._test_datasets: list[IterableDataset] = []

    def _get_dataset(self, name: str, split: Split) -> Dataset:
        """Get a dataset instance with appropriate tokenizer transform."""
        dataset_info = SUPPORTED_DATASETS_INFO[name]

        modality = dataset_info.modality
        transform = self._tokenizer_transforms[modality]
        dataset_class = dataset_info.dataset_class

        match name:
            case "M320M":
                dataset = dataset_class(
                    root=self._root,
                    transform=transform,
                    keys=["smiles"],
                    split=split.value,
                    shuffle=(split == Split.TRAIN),
                    download=True,
                    shuffle_buffer_size=self._shuffle_buffer_size,
                )
            case "Calm":
                dataset = dataset_class(
                    root=self._root,
                    transform=transform,
                    keys=["sequence"],
                    shuffle=(split == Split.TRAIN),
                    download=True,
                    shuffle_buffer_size=self._shuffle_buffer_size,
                )
            case "AMPLIFY":
                dataset = dataset_class(
                    root=self._root,
                    transform=transform,
                    download=False,
                    split=split.value,
                    shuffle=(split == Split.TRAIN),
                    shuffle_buffer_size=self._shuffle_buffer_size,
                )
            case _:
                raise ValueError(f"Dataset {name} is not supported")

        return dataset

    def setup(self, stage: str | None = None) -> None:
        self._train_datasets = []
        self._val_datasets = []

        for dataset_name in self.dataset_names:
            dataset_info = SUPPORTED_DATASETS_INFO[dataset_name]

            if Split.TRAIN in dataset_info.supported_splits:
                train_dataset = self._get_dataset(dataset_name, split=Split.TRAIN)
                self._train_datasets.append(train_dataset)

            # Combine test and validation datasets into a single validation set
            # (datasets name their splits differently, for testing, we'll use completely separate datasets)
            if Split.VALIDATION in dataset_info.supported_splits:
                val_dataset = self._get_dataset(dataset_name, split=Split.VALIDATION)
                self._val_datasets.append(val_dataset)

            if Split.TEST in dataset_info.supported_splits:
                test_dataset = self._get_dataset(dataset_name, split=Split.TEST)
                self._val_datasets.append(test_dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            ChainDataset(self._train_datasets),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        return DataLoader(
            ChainDataset(self._val_datasets),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        )
