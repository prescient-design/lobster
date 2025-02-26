from functools import partial
from pathlib import Path
from typing import Sequence

from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader

from lobster.constants import Modality

try:
    from nodehaha.collate import Collator
    from sethaha.sources import AMPLIFYDataset, CalmDataset, M320MDataset
    from sethaha.transforms.text import TokenizeNode
    from torchdata.nodes import BaseNode, Batcher, Loader, MultiNodeWeightedSampler, PinMemory

    HAHA = True

except (ImportError, ModuleNotFoundError):
    HAHA = False


# Updated dictionary to include supported splits for each dataset
SUPPORTED_DATASETS_INFO = {
    "M320M": {"modality": Modality.SMILES, "supported_splits": ["train", "validation", "test"]},
    "Calm": {"modality": Modality.NUCLEOTIDE, "supported_splits": ["train"]},
    "AMPLIFY": {"modality": Modality.AMINO_ACID, "supported_splits": ["train", "test"]},
}


class UmeHahaLightningDataModule(LightningDataModule):
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
        shuffle_buffer_size: int = 1000,
    ) -> None:
        if not HAHA:
            raise ImportError("Haha! Error! The datahaha package is required to use this data module. ")
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
        self._shuffle_buffer_size = shuffle_buffer_size

        # Initialize tokenizers for each modality
        self._tokenize_nodes = {
            Modality.SMILES: partial(
                TokenizeNode,
                tokenizer_name="karina-zadorozhny/smiles_tokenizer",
                return_tensors="pt",
                padding="max_length",
                max_length=self._tokenizer_max_length,
                truncation=True,
            ),
            Modality.AMINO_ACID: partial(
                TokenizeNode,
                tokenizer_name="karina-zadorozhny/amino_acid_tokenizer",
                return_tensors="pt",
                padding="max_length",
                max_length=self._tokenizer_max_length,
                truncation=True,
            ),
            Modality.NUCLEOTIDE: partial(
                TokenizeNode,
                tokenizer_name="karina-zadorozhny/nucleotide_tokenizer",
                return_tensors="pt",
                padding="max_length",
                max_length=self._tokenizer_max_length,
                truncation=True,
            ),
        }

        self._train_nodes = {}
        self._val_nodes = {}

    def _get_tokenized_dataset_node(self, name: str, split: str):
        dataset_info = SUPPORTED_DATASETS_INFO[name]
        modality = dataset_info["modality"]
        partial_tokenize_node = self._tokenize_nodes[modality]

        # Check if the requested split is supported
        if split not in dataset_info["supported_splits"]:
            raise ValueError(f"Split '{split}' is not supported for dataset '{name}'")

        match name:
            case "M320M":
                dataset_class = M320MDataset
                text_key = "smiles"
            case "Calm":
                dataset_class = CalmDataset
                text_key = "sequence"
            case "AMPLIFY":
                dataset_class = AMPLIFYDataset
                text_key = "sequence"
            case _:
                raise ValueError(f"Dataset {name} is not supported")

        dataset = dataset_class(
            split=split,
            download=True,
            streaming=True,
            cache_dir=self._root,
            example_shuffle_buffer=self._shuffle_buffer_size if split == "train" else None,
        )

        node = dataset.create_node()
        node = partial_tokenize_node(source_node=node, text_key=text_key)

        return node

    def setup(self, stage: str | None = None) -> None:
        # Initialize all dataset nodes regardless of stage

        for dataset_name in self.dataset_names:
            dataset_info = SUPPORTED_DATASETS_INFO[dataset_name]
            supported_splits = dataset_info["supported_splits"]

            if "train" in supported_splits:
                train_node = self._get_tokenized_dataset_node(dataset_name, "train")
                self._train_nodes[dataset_name] = train_node

            if "validation" in supported_splits:
                val_nodes = self._get_tokenized_dataset_node(dataset_name, "validation")
                self._val_nodes[dataset_name] = val_nodes

            if "test" in supported_splits:
                val_nodes = self._get_tokenized_dataset_node(dataset_name, "test")
                self._val_nodes[dataset_name] = val_nodes

    def _get_loader(self, dataset_nodes: dict[str, BaseNode]) -> Loader | None:
        node = MultiNodeWeightedSampler(
            dataset_nodes,
            # TODO add support for weights
            weights={key: 1 / len(dataset_nodes) for key in dataset_nodes},
        )
        node = Batcher(node, batch_size=self._batch_size)
        node = Collator(node)

        if self._pin_memory:
            node = PinMemory(node)

        # # node = Prefetcher(node, prefetch_factor=2)

        return Loader(node, restart_on_stop_iteration=True)

    def train_dataloader(self) -> DataLoader:
        return self._get_loader(self._train_nodes)

    def val_dataloader(self) -> DataLoader | None:
        return self._get_loader(self._val_nodes)
