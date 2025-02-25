from pathlib import Path
from typing import Sequence, TypeVar

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import ChainDataset, DataLoader, Dataset, IterableDataset

from lobster.constants import Modality
from lobster.datasets import CalmIterableDataset, M320MIterableDataset
from lobster.tokenization import AminoAcidTokenizerFast, NucleotideTokenizerFast, SmilesTokenizerFast
from lobster.transforms import TokenizerTransform

T = TypeVar("T")


# Updated dictionary to include supported splits for each dataset
SUPPORTED_DATASETS_INFO = {
    "M320M": {"modality": Modality.SMILES, "supported_splits": ["train", "validation", "test"]},
    "Calm": {"modality": Modality.NUCLEOTIDE, "supported_splits": ["train"]},
}


class UmeLightningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        tokenizer_max_length: int,
        datasets: None | Sequence[str] = None,
        root: Path | str | None = None,
        use_text_descriptions: bool = False,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
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
        self._use_text_descriptions = use_text_descriptions
        self._generator = Generator().manual_seed(seed)
        self._seed = seed
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._drop_last = drop_last

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

        self._train_datasets: list[IterableDataset] = []
        self._val_datasets: list[IterableDataset] = []
        self._test_datasets: list[IterableDataset] = []

        # Optional: Add a collate function if needed
        self._collate_fn = None

    def _get_dataset(self, name: str, split: str) -> Dataset:
        """Get a dataset instance with appropriate tokenizer transform."""
        dataset_info = SUPPORTED_DATASETS_INFO[name]
        modality = dataset_info["modality"]
        transform = self._tokenizer_transforms[modality]

        # Check if the requested split is supported
        if split not in dataset_info["supported_splits"]:
            raise ValueError(f"Split '{split}' is not supported for dataset '{name}'")

        match name:
            case "M320M":
                return M320MIterableDataset(
                    root=self._root,
                    transform=transform,
                    keys=["smiles", "Description"] if self._use_text_descriptions else ["smiles"],
                    split=split,
                )
            case "Calm":
                if split != "train":
                    raise ValueError("Calm dataset only supports training split")

                return CalmIterableDataset(
                    root=self._root,
                    transform=transform,
                    keys=["sequence", "description"] if self._use_text_descriptions else ["sequence"],
                )
            case _:
                raise ValueError(f"Dataset {name} is not supported")

    def setup(self, stage: str | None = None) -> None:
        # Initialize all datasets regardless of stage
        self._train_datasets = []
        self._val_datasets = []
        self._test_datasets = []

        for dataset_name in self.dataset_names:
            dataset_info = SUPPORTED_DATASETS_INFO[dataset_name]
            supported_splits = dataset_info["supported_splits"]

            if "train" in supported_splits:
                train_dataset = self._get_dataset(dataset_name, "train")
                self._train_datasets.append(train_dataset)

            if "validation" in supported_splits:
                val_dataset = self._get_dataset(dataset_name, "validation")
                self._val_datasets.append(val_dataset)

            if "test" in supported_splits:
                test_dataset = self._get_dataset(dataset_name, "test")
                self._test_datasets.append(test_dataset)

    def train_dataloader(self) -> DataLoader:
        if not self._train_datasets:
            raise ValueError("No training datasets available")

        return DataLoader(
            ChainDataset(self._train_datasets),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> DataLoader | None:
        if not self._val_datasets:
            return None

        return DataLoader(
            ChainDataset(self._val_datasets),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    def test_dataloader(self) -> DataLoader | None:
        if not self._test_datasets:
            return None

        return DataLoader(
            ChainDataset(self._test_datasets),
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
