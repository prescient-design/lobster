from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import pooch
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

from lobster.constants import (
    CALM_TASK_SPECIES,
    CALM_TASKS,
    CALMSpecies,
    CALMTask,
)
from lobster.transforms import Transform


class CalmPropertyDataset(Dataset):
    """Dataset for CALM property prediction tasks.

    This dataset provides access to various biological sequence property prediction tasks from
    the CALM dataset collection, supporting both regression and multilabel classification tasks.
    Data is loaded from Hugging Face with local caching.

    Parameters
    ----------
    task : Task | str
        The CALM task to load data for. Available tasks include:
        - 'meltome': Predicts protein melting temperature (regression)
        - 'solubility': Predicts protein solubility (regression)
        - 'localization': Predicts cellular localization (multilabel, 10 classes)
        - 'protein_abundance': Predicts protein abundance (regression, species-specific)
        - 'transcript_abundance': Predicts transcript abundance (regression, species-specific)
        - 'function_bp': Predicts Gene Ontology biological process terms (multilabel, 4 classes)
        - 'function_cc': Predicts Gene Ontology cellular component terms (multilabel, 4 classes)
        - 'function_mf': Predicts Gene Ontology molecular function terms (multilabel, 4 classes)
    root : str | Path | None, default=None
        Root directory for data storage. If None, uses system cache.
    species : Optional[Species | str], default=None
        Species for species-specific tasks. Required for 'protein_abundance' and
        'transcript_abundance' tasks.

        Available species per task:
        - For protein_abundance: 'athaliana', 'dmelanogaster', 'ecoli', 'hsapiens', 'scerevisiae'
        - For transcript_abundance: All of the above plus 'hvolcanii' and 'ppastoris'

        Not applicable for other tasks.
    split : Optional[Literal["train", "validation", "test"]], default=None
        Data split to use (if available). Currently ignored as all data is in a single split.
    download : bool, default=True
        Whether to download data if not available locally.
    transform_fn : Optional[Callable | Transform], default=None
        Transformation to apply to input sequences.
    target_transform_fn : Optional[Callable | Transform], default=None
        Transformation to apply to target values.
    columns : Optional[Sequence[str]], default=None
        Specific columns to use from the dataset. If None, appropriate columns are
        selected based on the task.
    force_download : bool, default=False
        If True, forces re-download from Hugging Face even if local cache exists.
    """

    def __init__(
        self,
        task: CALMTask | str,
        root: str | Path | None = None,
        *,
        species: CALMSpecies | str | None = None,
        split: Literal["train", "validation", "test"] | None = None,
        download: bool = True,
        transform_fn: Callable | Transform | None = None,
        target_transform_fn: Callable | Transform | None = None,
        columns: Sequence[str] | None = None,
        force_download: bool = False,
    ):
        super().__init__()

        if isinstance(task, str):
            task = CALMTask(task)
        if isinstance(species, str):
            species = CALMSpecies(species)

        self.task = task
        self.species = species
        self.split = split
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn

        self.task_type, self.num_classes = CALM_TASKS[task.value]

        if root is None:
            root = pooch.os_cache("lbster")
        if isinstance(root, str):
            root = Path(root)
        self.root = root.resolve()

        # Configure paths based on task and species
        self.hf_data_file, self.cache_path = self._configure_paths()

        # Load data from Hugging Face or cache
        self._load_data("taylor-joren/calm-property", download, force_download)

        self._set_columns(columns)

    def _configure_paths(self) -> tuple[str, Path]:
        """Configure file paths based on task and species.

        Returns
        -------
        Tuple[str, Path]
            A tuple containing (huggingface_data_file, local_cache_path)
        """
        task = self.task
        species = self.species

        # Get the appropriate file path in the HF repo
        if task in [CALMTask.FUNCTION_BP, CALMTask.FUNCTION_CC, CALMTask.FUNCTION_MF]:
            function_type = task.value.split("_")[1].lower()
            filename = f"calm_GO_{function_type}_middle_normal.parquet"
            hf_data_file = f"{task.value}/{filename}"

        elif task in [CALMTask.MELTOME, CALMTask.SOLUBILITY, CALMTask.LOCALIZATION]:
            filename = f"{task.value}.parquet"
            hf_data_file = f"{task.value}/{filename}"

        elif task in [CALMTask.PROTEIN_ABUNDANCE, CALMTask.TRANSCRIPT_ABUNDANCE]:
            if species is None:
                raise ValueError(f"Must specify species for {task.value} task")
            if species not in CALM_TASK_SPECIES[task]:
                raise ValueError(f"Species {species.value} not available for {task.value} task")

            filename = f"{task.value}_{species.value}.parquet"
            hf_data_file = f"{task.value}/{filename}"

        # Local cache path
        cache_path = self.root / self.__class__.__name__ / filename

        return hf_data_file, cache_path

    def _load_data(self, huggingface_repo: str, download: bool, force_download: bool) -> None:
        """Load data from Hugging Face or local cache.

        Parameters
        ----------
        huggingface_repo : str
            Hugging Face repository ID
        download : bool
            Whether to download data if not found locally
        force_download : bool
            Force re-download even if local cache exists
        """
        # Check if we need to download
        need_download = force_download or not self.cache_path.exists()

        if need_download and download:
            try:
                # Create parent directory if it doesn't exist
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)

                # Load from Hugging Face
                dataset = load_dataset(huggingface_repo, data_files=self.hf_data_file, split="train")

                # Convert to pandas and save to cache
                df = dataset.to_pandas()
                df.to_parquet(self.cache_path, index=False)

                # Use the dataframe
                self.data = df

            except Exception as e:
                print(f"Failed to load dataset from Hugging Face: {e}")
                raise

        elif self.cache_path.exists():
            # Load from local cache
            self.data = pd.read_parquet(self.cache_path)

        else:
            raise FileNotFoundError(f"Dataset file {self.cache_path} not found locally and download=False")

    def _set_columns(self, columns=None):
        """Set the columns to use based on the task."""
        if columns is None:
            match self.task:
                case CALMTask.FUNCTION_BP:
                    columns = ["sequence", "GO:0051092", "GO:0016573", "GO:0031146", "GO:0071427", "GO:0006613"]
                case CALMTask.FUNCTION_CC:
                    columns = ["sequence", "GO:0022627", "GO:0000502", "GO:0034705", "GO:0030665", "GO:0005925"]
                case CALMTask.FUNCTION_MF:
                    columns = ["sequence", "GO:0004843", "GO:0004714", "GO:0003774", "GO:0008227", "GO:0004866"]
                case CALMTask.LOCALIZATION:
                    columns = [
                        "Sequence",
                        "Cell membrane",
                        "Cytoplasm",
                        "Endoplasmic reticulum",
                        "Extracellular",
                        "Golgi apparatus",
                        "Lysosome/Vacuole",
                        "Mitochondrion",
                        "Nucleus",
                        "Peroxisome",
                        "Plastid",
                    ]
                case CALMTask.MELTOME:
                    columns = ["sequence", "melting_temperature"]
                case CALMTask.SOLUBILITY:
                    columns = ["cds", "solubility"]
                case CALMTask.PROTEIN_ABUNDANCE:
                    columns = ["cds", "abundance"]
                case CALMTask.TRANSCRIPT_ABUNDANCE:
                    columns = ["cds", "logtpm"]
                case _:
                    columns = list(self.data.columns)

        # Verify that all columns exist in the dataset
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in dataset. Available columns: {list(self.data.columns)}")

        self.columns = columns

    def __getitem__(self, index: int) -> tuple[str | Tensor, Tensor]:
        item = self.data.iloc[index]

        x = item[self.columns[0]]  # First column is always the input sequence/data

        if self.transform_fn is not None:
            x = self.transform_fn(x)

        y_cols = self.columns[1:]
        y_values = pd.to_numeric(item[y_cols]).values

        if self.task_type == "regression":
            y = torch.tensor(y_values, dtype=torch.float32)
        else:  # multilabel tasks (localization and function prediction)
            y = torch.tensor(y_values, dtype=torch.long)

        if self.target_transform_fn is not None:
            y = self.target_transform_fn(y)

        return x, y

    def __len__(self) -> int:
        return len(self.data)
