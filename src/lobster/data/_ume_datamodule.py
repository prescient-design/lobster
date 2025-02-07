from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence, TypeVar

from beignet.datasets import ChEMBLDataset
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Sampler

from lobster.datasets import CalmDataset, M320MDataset, MultiplexedDataset
from lobster.tokenization import NucleotideTokenizerFast, SmilesTokenizerFast
from lobster.transforms import TokenizerTransform

T = TypeVar("T")

SUPPORTED_DATASETS = {"M320M", "ChEMBL", "Calm"}


class UmeLightningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        tokenizer_max_length: int,
        datasets: Sequence[str] | Dict[str, float] = None,
        root: Path | str | None = None,
        download: bool = False,
        use_text_descriptions: bool = False,
        generator: Generator | None = None,
        seed: int = 0xDEADBEEF,
        batch_size: int = 1,
        batch_sampler: Iterable[Sequence] | Sampler[Sequence] | None = None,
        num_workers: int = 0,
        collate_fn: Callable[[list[T]], Any] | None = None,
        max_length: int = 512,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()

        if datasets is None:
            self.datasets = SUPPORTED_DATASETS
            self.weights = None
        elif isinstance(datasets, dict):
            self.datasets = set(datasets.keys())
            self.weights = list(datasets.values())
        else:
            self.datasets = set(datasets)
            self.weights = None

        if not self.datasets.issubset(SUPPORTED_DATASETS):
            raise ValueError(
                f"Only the following datasets are supported: {SUPPORTED_DATASETS}."
                f"Unknown datasets: {self.datasets - SUPPORTED_DATASETS}"
            )

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._root = root
        self._download = download
        self._tokenizer_max_length = tokenizer_max_length
        self._use_text_descriptions = use_text_descriptions
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._max_length = max_length
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last

    def prepare_data(self) -> None:
        smiles_transform = TokenizerTransform(
            SmilesTokenizerFast(), padding="max_length", truncation=True, max_length=self._tokenizer_max_length
        )
        nucleotide_transform = TokenizerTransform(
            NucleotideTokenizerFast(), padding="max_length", truncation=True, max_length=self._tokenizer_max_length
        )

        dataset_instances = []

        if "M320M" in self.datasets:
            dataset_instances.append(
                M320MDataset(
                    root=self._root,
                    download=self._download,
                    transform=smiles_transform,
                    columns=["smiles", "Description"] if self._use_text_descriptions else ["smiles"],
                )
            )

        if "ChEMBL" in self.datasets:
            dataset_instances.append(
                ChEMBLDataset(root=self._root, download=self._download, transform=smiles_transform)
            )

        if "Calm" in self.datasets:
            dataset_instances.append(
                CalmDataset(
                    root=self._root,
                    transform=nucleotide_transform,
                    columns=["sequence", "description"] if self._use_text_descriptions else ["sequence"],
                )
            )

        self.dataset = MultiplexedDataset(
            datasets=dataset_instances,
            weights=self.weights,
            seed=self._seed,
        )

    def setup(self, stage: str = "fit") -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )

    # TODO zadorozk: Implement validation dataset
    # The easiest way would be to have separate datasets
    # for each modality for validation and testing
    # since IterableDataset does not support random
    # splitting
    def val_dataloader(self) -> Any:
        raise NotImplementedError()
