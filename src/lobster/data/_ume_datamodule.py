from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Type

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, IterableDataset

from lobster.constants import Modality, Split
from lobster.datasets import (
    AMPLIFYIterableDataset,
    CalmIterableDataset,
    HuggingFaceIterableDataset,
    M320MIterableDataset,
    MultiplexedSamplingDataset,
)
from lobster.tokenization import AminoAcidTokenizerFast, NucleotideTokenizerFast, SmilesTokenizerFast
from lobster.transforms import TokenizerTransform


@dataclass
class DatasetInfo:
    name: str
    dataset_class: Type
    modality: Modality
    supported_splits: set[Split]
    train_size: int  # Approximate size of the training set, used for sampling weights
    test_size: int | None = None  # Approximate size of the test set, used for sampling weights
    kwargs: dict[str, any] = None

    def __post_init__(self):
        if not issubclass(self.dataset_class, HuggingFaceIterableDataset):
            raise ValueError("Only HuggingFaceIterableDataset subclasses are currently supported.")

        if Split.TEST in self.supported_splits and self.test_size is None:
            raise ValueError("test_size must be provided if Split.TEST is supported.")


SUPPORTED_DATASETS_INFO = [
    DatasetInfo(
        name="M320M",
        dataset_class=M320MIterableDataset,
        modality=Modality.SMILES,
        supported_splits={Split.TRAIN, Split.TEST},
        train_size=19_400_000,
        test_size=1_000_000,
        kwargs={"download": False, "keys": ["smiles"]},
    ),
    DatasetInfo(
        name="Calm",
        dataset_class=CalmIterableDataset,
        modality=Modality.NUCLEOTIDE,
        supported_splits={Split.TRAIN},
        train_size=8_780_000,
        kwargs={"download": False, "keys": ["sequence"]},
    ),
    DatasetInfo(
        name="AMPLIFY",
        dataset_class=AMPLIFYIterableDataset,
        modality=Modality.AMINO_ACID,
        supported_splits={Split.TRAIN, Split.TEST},
        train_size=448_000_000,
        test_size=40_000,
        kwargs={"download": False},
    ),
]


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

        supported_datasets = {info.name for info in SUPPORTED_DATASETS_INFO}

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

        self._train_sizes: list[int] = []
        self._val_sizes: list[int] = []
        self._test_sizes: list[int] = []

        self.train_dataset: MultiplexedSamplingDataset | None = None
        self.val_dataset: MultiplexedSamplingDataset | None = None

    def _get_dataset(self, dataset_info: DatasetInfo, split: Split) -> Dataset:
        """Get a dataset instance with appropriate tokenizer transform."""

        dataset_class = dataset_info.dataset_class
        transform = self._tokenizer_transforms[dataset_info.modality]

        return dataset_class(
            root=self._root,
            transform=transform,
            split=split.value,
            shuffle=(split == Split.TRAIN),
            shuffle_buffer_size=self._shuffle_buffer_size,
            **dataset_info.kwargs,
        )

    def setup(self, stage: str | None = None) -> None:
        self._train_datasets = []
        self._val_datasets = []

        self._train_sizes = []
        self._val_sizes = []

        for dataset_name in self.dataset_names:
            dataset_info = next(info for info in SUPPORTED_DATASETS_INFO if info.name == dataset_name)

            if Split.TRAIN in dataset_info.supported_splits:
                train_dataset = self._get_dataset(dataset_info, split=Split.TRAIN)

                self._train_datasets.append(train_dataset)
                self._train_sizes.append(dataset_info.train_size)

            if Split.TEST in dataset_info.supported_splits:
                val_dataset = self._get_dataset(dataset_info, split=Split.TEST)

                self._val_datasets.append(val_dataset)
                self._val_sizes.append(dataset_info.test_size)

        self.train_dataset = MultiplexedSamplingDataset(self._train_datasets, seed=self._seed, mode="min")
        self.val_dataset = MultiplexedSamplingDataset(
            self._val_datasets,
            mode="min",
            seed=self._seed,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        )
