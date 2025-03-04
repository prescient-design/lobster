from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Type

from lightning import LightningDataModule
from nodehaha.collate import Collator
from nodehaha.round_robin_node import RoundRobinNode
from sethaha.sources import AMPLIFYDataset, CalmDataset, M320MDataset
from sethaha.sources.base import BaseHuggingFaceDataset
from sethaha.transforms.text import TokenizeNode
from torch import Generator
from torch.utils.data import DataLoader, Dataset
from torchdata.nodes import BaseNode, Batcher, Loader, PinMemory

from lobster.constants import Modality, Split


@dataclass
class DatasetInfo:
    name: str
    text_key: str
    dataset_class: Type
    modality: Modality
    supported_splits: set[Split]
    train_size: int  # Approximate size of the training set, used for sampling weights
    test_size: int | None = None  # Approximate size of the test set, used for sampling weights
    kwargs: dict[str, any] = None

    def __post_init__(self):
        if not issubclass(self.dataset_class, BaseHuggingFaceDataset):
            raise NotImplementedError("Only BaseHuggingFaceDataset subclasses are currently supported.")


SUPPORTED_DATASETS_INFO = [
    DatasetInfo(
        name="M320M",
        dataset_class=M320MDataset,
        text_key="smiles",
        modality=Modality.SMILES,
        supported_splits={Split.TRAIN, Split.TEST},
        train_size=19_400_000,
        test_size=1_000_000,
    ),
    DatasetInfo(
        name="Calm",
        dataset_class=CalmDataset,
        text_key="sequence",
        modality=Modality.NUCLEOTIDE,
        supported_splits={Split.TRAIN},  # TODO: add splits
        train_size=8_780_000,
    ),
    DatasetInfo(
        name="AMPLIFY",
        dataset_class=AMPLIFYDataset,
        text_key="sequence",
        modality=Modality.AMINO_ACID,
        supported_splits={Split.TRAIN, Split.TEST},
        train_size=448_000_000,
        test_size=40_000,
    ),
]


class UmeHahaLightningDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        tokenizer_max_length: int,
        datasets: None | Sequence[str] = None,
        root: Path | str | None = None,
        seed: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
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
        self.tokenizer_names = {
            Modality.SMILES: "karina-zadorozhny/smiles_tokenizer",
            Modality.AMINO_ACID: "karina-zadorozhny/amino_acid_tokenizer",
            Modality.NUCLEOTIDE: "karina-zadorozhny/nucleotide_tokenizer",
        }

        self._train_nodes: list[BaseNode] = []
        self._val_nodes: list[BaseNode] = []

        self.train_node: BaseNode | None = None
        self.val_node: BaseNode | None = None

    def _get_dataset_node(self, dataset_info: DatasetInfo, split: Split) -> Dataset:
        """Get a dataset instance with appropriate tokenizer transform."""

        dataset_class = dataset_info.dataset_class

        dataset = dataset_class(
            split=split.value,
            example_shuffle_buffer=self._shuffle_buffer_size if split == Split.TRAIN else 1,
        )
        node = dataset.create_node()

        node = TokenizeNode(
            node,
            tokenizer_name=self.tokenizer_names[dataset_info.modality],
            text_key="sequence",
            return_tensors="pt",
            padding="max_length",
            max_length=self._tokenizer_max_length,
        )

        return node

    def setup(self, stage: str | None = None) -> None:
        self._train_nodes = []
        self._val_nodes = []

        for dataset_name in self.dataset_names:
            dataset_info = next(info for info in SUPPORTED_DATASETS_INFO if info.name == dataset_name)

            if Split.TRAIN in dataset_info.supported_splits:
                train_node = self._get_dataset_node(dataset_info, split=Split.TRAIN)
                self._train_nodes.append(train_node)

            # Uses test sets for validation
            # This assumes that we're going to test on completely different datasets
            # TODO: standardize train,val, test in dataset creation pipeline
            if Split.TEST in dataset_info.supported_splits:
                val_node = self._get_dataset_node(dataset_info, split=Split.TEST)
                self._val_nodes.append(val_node)

        self.train_node = RoundRobinNode(self._train_nodes)
        self.val_node = RoundRobinNode(self._val_nodes)

    def _get_dataloader(self, node):
        node = Batcher(node, batch_size=self._batch_size)
        node = Collator(node)

        if self._pin_memory:
            node = PinMemory(node)

        return Loader(node)

    def train_dataloader(self) -> DataLoader:
        if self.train_node is None:
            raise ValueError("DataModule must be setup before calling train_dataloader")

        return self._get_dataloader(self.train_node)

    def val_dataloader(self) -> DataLoader | None:
        if self.val_node is None:
            raise ValueError("DataModule must be setup before calling val_dataloader")

        return self._get_dataloader(self.val_node)
