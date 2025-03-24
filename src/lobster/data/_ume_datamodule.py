from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Type

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.transforms import Compose, Lambda

from lobster.constants import Modality, Split
from lobster.datasets import (
    AMPLIFYIterableDataset,
    CalmIterableDataset,
    HuggingFaceIterableDataset,
    LatentGeneratorPinderIterableDataset,
    M320MIterableDataset,
    MultiplexedSamplingDataset,
    OpenGenome2IterableDataset,
    RoundRobinConcatIterableDataset,
)
from lobster.tokenization import (
    UmeAminoAcidTokenizerFast,
    UmeLatentGenerator3DCoordTokenizerFast,
    UmeNucleotideTokenizerFast,
    UmeSmilesTokenizerFast,
)
from lobster.transforms import TokenizerTransform
from lobster.transforms.functional import sample_tokenized_input


@dataclass
class DatasetInfo:
    name: str
    dataset_class: Type
    modality: Modality
    supported_splits: set[Split]
    train_size: int  # Approximate size of the training set, used for sampling weights
    test_size: int | None = None  # Approximate size of the test set, used for sampling weights
    kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if not issubclass(self.dataset_class, HuggingFaceIterableDataset):
            raise NotImplementedError("Only HuggingFaceIterableDataset subclasses are currently supported.")


SUPPORTED_DATASETS_INFO = [
    DatasetInfo(
        name="M320M",
        dataset_class=M320MIterableDataset,
        modality=Modality.SMILES,
        supported_splits={Split.TRAIN, Split.TEST},
        train_size=19_400_000,
        test_size=1_000_000,
        kwargs={"keys": ["smiles"]},
    ),
    DatasetInfo(
        name="Calm",
        dataset_class=CalmIterableDataset,
        modality=Modality.NUCLEOTIDE,
        supported_splits={Split.TRAIN},  # TODO: add splits
        train_size=8_780_000,  # NOTE - this is an underestimate (whole genomes much longer)
        kwargs={"keys": ["sequence"]},
    ),
    DatasetInfo(
        name="AMPLIFY",
        dataset_class=AMPLIFYIterableDataset,
        modality=Modality.AMINO_ACID,
        supported_splits={Split.TRAIN, Split.TEST},
        train_size=448_000_000,
        test_size=40_000,
    ),
    DatasetInfo(
        name="Pinder",
        dataset_class=LatentGeneratorPinderIterableDataset,
        modality=Modality.COORDINATES_3D,
        supported_splits={Split.TRAIN, Split.TEST},
        train_size=267_000,
        test_size=2_000,
    ),
    DatasetInfo(
        name="OpenGenome2",
        dataset_class=OpenGenome2IterableDataset,
        modality=Modality.NUCLEOTIDE,
        supported_splits={Split.TRAIN, Split.VALIDATION, Split.TEST},
        train_size=28_840_000_000,  # 8.84T nucleotides, computed relative to AMPLIFY's 138B tokens
        test_size=100_000,  # Placeholder value for test/validation set
        kwargs={"keys": ["text"]},
    ),
]


class UmeLightningDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer_max_length: int,
        *,
        datasets: None | Sequence[str] = None,
        download: bool = False,
        root: Path | str | None = None,
        seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 10000,
        stopping_condition: str = "min",
        sample: bool = False,
        weights: Sequence[float | int] | None = None,
    ) -> None:
        """Initialize a UmeLightningDataModule.

        Parameters
        ----------
        tokenizer_max_length : int
            Maximum length of the tokenized input. Should match the model's maximum input length.
        datasets : None | Sequence[str], optional
            List of dataset names to use. If None, all supported datasets will be used.
            Example: ["M320M", "Calm", "AMPLIFY", "Pinder"]
        download: bool, optional
            If True, will download the datasets first and stream locally.
            Otherwise, streams directly from Hugging Face.
            Downloaded datasets are cached in the `root` directory.
        root : Path | str | None, optional
            Root directory where the datasets are stored. If None, the default directory will be used.
        seed : int, optional
            Random seed for reproducibility.
        batch_size : int, optional
            Batch size.
        num_workers : int, optional
            Number of workers for data loading.
        pin_memory : bool, optional
            Whether to pin memory for faster GPU transfer.
        shuffle_buffer_size : int, optional
            Size of the shuffle buffer for training datasets. Is for shuffling iterable datasets.
        stopping_condition : str, optional
            Stopping condition for `RoundRobinConcatIterableDataset`. Can be "min" or "max".
            If min, the dataset will stop when the smallest dataset is exhausted.
            If max, the dataset will stop when the largest dataset is exhausted.
            Ignored if sample is True.
        sample : bool, optional
            Whether to sample from the datasets with replacement with `MultiplexedSamplingDataset`.
            If True, `MultiplexedSamplingDataset` is used (with optional `weights` parameter)
            and `stopping_condition` is ignored.
            If False, `RoundRobinConcatIterableDataset` is used.
        weights : Sequence[float | int] | None, optional
            Sampling weights for the datasets.
            If None, uses dataset sizes as weights to ensure sampling proportional to dataset size.
            If you want to sample uniformly from all datasets, set all weights to 1.0.
            Ignored if sample is False.

        """
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
        self._stopping_condition = stopping_condition
        self._sample = sample
        self._weights = weights
        self._download = download

        # Initialize tokenizer transforms for each modality
        tokenizer_instances = {
            Modality.SMILES: UmeSmilesTokenizerFast(),
            Modality.AMINO_ACID: UmeAminoAcidTokenizerFast(),
            Modality.NUCLEOTIDE: UmeNucleotideTokenizerFast(),
            Modality.COORDINATES_3D: UmeLatentGenerator3DCoordTokenizerFast(),
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

        self.train_dataset: RoundRobinConcatIterableDataset | MultiplexedSamplingDataset | None = None
        self.val_dataset: RoundRobinConcatIterableDataset | MultiplexedSamplingDataset | None = None

    def _get_dataset(self, dataset_info: DatasetInfo, split: Split) -> Dataset:
        """Get a dataset instance with appropriate tokenizer transform."""

        dataset_class = dataset_info.dataset_class
        transform = self._tokenizer_transforms[dataset_info.modality]

        if dataset_info.modality == Modality.COORDINATES_3D:
            # Sample 1 pose out of 4 for the latent generator datasets
            transform = Compose([transform, Lambda(sample_tokenized_input)])

        return dataset_class(
            root=self._root,
            download=self._download,
            transform=transform,
            split=split.value,
            shuffle=(split == Split.TRAIN),
            shuffle_buffer_size=self._shuffle_buffer_size,
            **dataset_info.kwargs or {},
        )

    def _get_aggregated_dataset(
        self, datasets: list[IterableDataset], sizes: list[int]
    ) -> MultiplexedSamplingDataset | RoundRobinConcatIterableDataset:
        if self._sample:
            weights = self._weights if self._weights is not None else sizes
            return MultiplexedSamplingDataset(
                datasets,
                weights=weights,
                seed=self._seed,
            )
        else:
            return RoundRobinConcatIterableDataset(
                datasets,
                stopping_condition=self._stopping_condition,
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

            # Uses test sets for validation
            # This assumes that we're going to test on completely different datasets
            # TODO: standardize train,val, test in dataset creation pipeline
            if Split.TEST in dataset_info.supported_splits:
                val_dataset = self._get_dataset(dataset_info, split=Split.TEST)

                self._val_datasets.append(val_dataset)
                self._val_sizes.append(dataset_info.test_size)

        self.train_dataset = self._get_aggregated_dataset(self._train_datasets, self._train_sizes)
        self.val_dataset = self._get_aggregated_dataset(self._val_datasets, self._val_sizes)

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
