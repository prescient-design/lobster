from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Sequence, TypeVar

from beignet.datasets import ChEMBLDataset
from lightning import LightningDataModule
from torch import Generator
from torch.utils.data import ChainDataset, DataLoader, Sampler

from lobster.datasets import (
    CalmDataset,
    DatasetToIterableDataset,
    FASTADataset,
    M320MDataset,
    MultiplexedSamplingDataset,
)
from lobster.tokenization import AminoAcidTokenizerFast, NucleotideTokenizerFast, SmilesTokenizerFast
from lobster.transforms import TokenizerTransform

T = TypeVar("T")

SUPPORTED_DATASETS = {"M320M", "ChEMBL", "Calm", "Uniref50"}


class UmeLightningDataModule(LightningDataModule):
    """Lightning DataModule for handling multiple molecular and biological sequence datasets.

    This DataModule supports multiple datasets including ChEMBL (molecules), M320M (molecules),
    Calm (nucleotide sequences), and Uniref50 (amino acid sequences). It can handle both
    regular dataset chaining and weighted sampling between datasets.

    Parameters
    ----------
    root : Path | str
        Root directory for dataset storage.
    tokenizer_max_length : int
        Maximum sequence length for tokenizer padding/truncation.
    datasets : None | Sequence[str] | Dict[str, float], optional
        Datasets to use. If None, uses all supported datasets.
        If dict, keys are dataset names and values are sampling weights.
        Supported datasets are: {M320M, ChEMBL, Calm, Uniref50}.
    enable_sampling : bool, default=False
        If True, enables weighted sampling between datasets.
        Required when providing dataset weights.
    download : bool, default=False
        If True, downloads datasets if not present.
    use_text_descriptions : bool, default=False
        If True, includes text descriptions in dataset outputs.
    generator : Generator | None, default=None
        PyTorch Generator for reproducibility. If None, creates new one with seed.
    seed : int, default=0xDEADBEEF
        Random seed for generator initialization.
    batch_size : int, default=1
        Number of samples per batch.
    batch_sampler : Iterable[Sequence] | Sampler[Sequence] | None, default=None
        Batch sampler for custom sampling strategies.
    num_workers : int, default=0
        Number of subprocesses for data loading.
    collate_fn : Callable[[list[T]], Any] | None, default=None
        Function to collate samples into batches.
    max_length : int, default=512
        Maximum sequence length for dataset processing.
    pin_memory : bool, default=True
        If True, pins memory in DataLoader.
    drop_last : bool, default=False
        If True, drops last non-full batch.

    Raises
    ------
    ValueError
        If unknown datasets are specified or weights provided without sampling enabled.
    """

    def __init__(
        self,
        *,
        root: Path | str,
        tokenizer_max_length: int,
        datasets: None | Sequence[str] | Dict[str, float] = None,
        enable_sampling: bool = False,
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
            self.dataset_names = SUPPORTED_DATASETS
            self.weights = None
        elif isinstance(datasets, dict):
            self.dataset_names = set(datasets.keys())
            self.weights = list(datasets.values())
        else:
            self.dataset_names = set(datasets)
            self.weights = None

        if not self.dataset_names.issubset(SUPPORTED_DATASETS):
            raise ValueError(
                f"Only the following datasets are supported: {SUPPORTED_DATASETS}."
                f"Unknown datasets: {self.dataset_names - SUPPORTED_DATASETS}"
            )
        if self.weights is not None and not enable_sampling:
            raise ValueError("Weights can only be provided if sampling behavior is enabled.")

        if generator is None:
            generator = Generator().manual_seed(seed)

        self._root = root
        self._download = download
        self._tokenizer_max_length = tokenizer_max_length
        self._use_text_descriptions = use_text_descriptions
        self._enable_sampling = enable_sampling
        self._generator = generator
        self._seed = seed
        self._batch_size = batch_size
        self._max_length = max_length
        self._batch_sampler = batch_sampler
        self._num_workers = num_workers
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._drop_last = drop_last

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

        self.datasets = []

        if "M320M" in self.dataset_names:
            self.datasets.append(
                DatasetToIterableDataset(
                    M320MDataset(
                        root=self._root,
                        download=self._download,
                        transform=smiles_transform,
                        columns=["smiles", "Description"] if self._use_text_descriptions else ["smiles"],
                    )
                )
            )

        if "ChEMBL" in self.dataset_names:
            self.datasets.append(
                DatasetToIterableDataset(
                    ChEMBLDataset(root=self._root, download=self._download, transform=smiles_transform)
                )
            )

        if "Calm" in self.dataset_names:
            self.datasets.append(
                DatasetToIterableDataset(
                    CalmDataset(
                        root=self._root,
                        transform=nucleotide_transform,
                        columns=["sequence", "description"] if self._use_text_descriptions else ["sequence"],
                    )
                )
            )
        if "Uniref50" in self.dataset_names:
            # TODO: Switch to AMPLIFY
            self.datasets.append(
                DatasetToIterableDataset(
                    FASTADataset(
                        root=Path(self._root) / "uniref50.fasta",  # TODO: FIXME
                        transform=amino_acid_transform,
                        use_text_descriptions=self._use_text_descriptions,
                    )
                )
            )

        if self._enable_sampling:
            self.dataset = MultiplexedSamplingDataset(
                datasets=self.datasets,
                weights=self.weights,
                seed=self._seed,
            )
        else:
            self.dataset = ChainDataset(self.datasets)

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
        return DataLoader(
            self.dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self._pin_memory,
        )
