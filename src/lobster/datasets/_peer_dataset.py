from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import pandas as pd
import pooch
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import Dataset

from lobster.constants import (
    PEER_TASK_CATEGORIES,
    PEER_TASK_COLUMNS,
    PEER_TASK_SPLITS,
    PEER_TASKS,
    PEERTask,
)


class PEERDataset(Dataset):
    """
    Datasets from PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding
    https://github.com/DeepGraphLearning/PEER_Benchmark

    Data splits are hosted on hugging face at taylor-joren/peer and locally cached after download
    --> originally available from torchdrug https://github.com/DeepGraphLearning/torchdrug/tree/master/torchdrug/datasets

    Parameters
    ----------
    task : PEERTask | str
        The PEER task to load data for. Available tasks span various categories:
        - Function prediction: 'aav', 'betalactamase', 'fluorescence', 'gb1', 'solubility', 'stability', 'thermostability'
        - Localization prediction: 'binarylocalization', 'subcellularlocalization'
        - Protein-ligand interaction: 'bindingdb', 'pdbbind'
        - Protein-protein interaction: 'humanppi', 'ppiaffinity', 'yeastppi'
        - Structure prediction: 'fold', 'proteinnet', 'secondarystructure'
    root : str | Path | None, default=None
        Root directory for data storage. If None, uses system cache.
    split : str, default="train"
        Data split to use. Available splits depend on the task, with most tasks having
        standard 'train', 'valid', and 'test' splits. Some tasks have special splits:
        - bindingdb: 'train', 'valid', 'random_test', 'holdout_test'
        - humanppi/yeastppi: 'train', 'valid', 'test', 'cross_species_test'
        - fold: 'train', 'valid', 'test_family_holdout', 'test_fold_holdout', 'test_superfamily_holdout'
        - secondarystructure: 'train', 'valid', 'casp12', 'cb513', 'ts115'
    download : bool, default=True
        Whether to download data if not available locally.
    transform_fn : Optional[Callable], default=None
        Transformation to apply to input sequences.
    target_transform_fn : Optional[Callable], default=None
        Transformation to apply to target values.
    columns : Optional[Sequence[str]], default=None
        Specific columns to use from the dataset. If None, appropriate columns are
        selected based on the task.
    force_download : bool, default=False
        If True, forces re-download from Hugging Face even if local cache exists.
    """

    def __init__(
        self,
        task: Union[PEERTask, str],
        root: Optional[Union[str, Path]] = None,
        *,
        split: str = "train",
        download: bool = True,
        transform_fn: Optional[Callable] = None,
        target_transform_fn: Optional[Callable] = None,
        columns: Optional[Sequence[str]] = None,
        force_download: bool = False,
    ):
        super().__init__()

        if isinstance(task, str):
            task = PEERTask(task)

        self.task = task
        self.split = split
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn

        if split not in PEER_TASK_SPLITS[task]:
            raise ValueError(f"Invalid split '{split}' for task '{task}'. Available splits: {PEER_TASK_SPLITS[task]}")

        self.task_type, self.num_classes = PEER_TASKS[task]
        self.task_category = PEER_TASK_CATEGORIES[task]

        if root is None:
            root = pooch.os_cache("lbster")
        if isinstance(root, str):
            root = Path(root)
        self.root = root.resolve()

        self.hf_data_file, self.cache_path = self._configure_paths()

        self._load_data("taylor-joren/peer", download, force_download)

        self._set_columns(columns)

    def _configure_paths(self) -> Tuple[str, Path]:
        """Configure file paths based on task and split.

        Returns
        -------
        Tuple[str, Path]
            A tuple containing (huggingface_data_file, local_cache_path)
        """
        task_category = self.task_category.value
        task = self.task.value
        split = self.split

        # Construct the file path in the HF repo
        filename = f"{split}.parquet"
        hf_data_file = f"{task_category}/{task}/{filename}"

        # Local cache path
        cache_path = self.root / self.__class__.__name__ / task_category / task / filename

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
        need_download = force_download or not self.cache_path.exists()

        if need_download and download:

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Load from Hugging Face
            dataset = load_dataset(huggingface_repo, data_files=self.hf_data_file, split="train")

            # Save to cache --> TODO - consider changing to polars
            df = dataset.to_pandas()
            df.to_parquet(self.cache_path, index=False)

            self.data = df

        elif self.cache_path.exists():
            # Load from local cache
            self.data = pd.read_parquet(self.cache_path)

        else:
            raise FileNotFoundError(f"Dataset file {self.cache_path} not found locally and download=False")

    def _set_columns(self, columns=None):
        """Set the columns to use based on the task using predefined column mappings."""
        if columns is None:
            if self.task not in PEER_TASK_COLUMNS:
                raise ValueError(f"No column mapping defined for task '{self.task}'")

            input_cols, target_cols = PEER_TASK_COLUMNS[self.task]
            columns = input_cols + target_cols

            # Verify that all expected columns exist in the dataset
            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Expected column '{col}' not found in dataset for task '{self.task}'. "
                        f"Available columns: {list(self.data.columns)}"
                    )

        self.columns = columns

    def __getitem__(self, index: int) -> Tuple[Union[str, Tensor], Tensor]:
        item = self.data.iloc[index]

        # Handle single input vs multiple inputs (like protein pairs)
        if self.task in PEER_TASK_COLUMNS:
            input_cols, _ = PEER_TASK_COLUMNS[self.task]
            if len(input_cols) == 1:
                # Single input case (most tasks)
                x = item[input_cols[0]]
                if self.transform_fn is not None:
                    x = self.transform_fn(x)
            else:
                # Multiple inputs case (protein pairs, protein-ligand)
                x = [item[col] for col in input_cols]
                if self.transform_fn is not None:
                    x = [self.transform_fn(xi) for xi in x]
        else:
            # Fallback: just use the first column as input
            x = item[self.columns[0]]
            if self.transform_fn is not None:
                x = self.transform_fn(x)

        # Get target values (all columns except the input columns)
        if self.task in PEER_TASK_COLUMNS:
            _, target_cols = PEER_TASK_COLUMNS[self.task]
        else:
            # Fallback: use all non-input columns as targets
            if isinstance(x, list):
                input_cols_count = len(x)
            else:
                input_cols_count = 1
            target_cols = self.columns[input_cols_count:]

        # Handle empty target_cols case (e.g., for unsupervised tasks)
        if not target_cols:
            y = torch.tensor([])
        else:
            # Convert to numeric, handling various potential data types
            try:
                y_values = pd.to_numeric(item[target_cols], errors="coerce").values
            except Exception:
                # If values are non-numeric, handle based on task_type
                if self.task_type in ["classification", "multilabel", "binary"]:
                    # For categorical data, use categorical encoding
                    y_values = item[target_cols].values
                else:
                    # For regression, try to force numeric
                    y_values = item[target_cols].astype(float).values

            # Create appropriate tensor based on task type
            if self.task_type == "regression":
                y = torch.tensor(y_values, dtype=torch.float32)
            elif self.task_type == "binary":
                y = torch.tensor(y_values, dtype=torch.float32)
            else:  # classification/multilabel
                y = torch.tensor(y_values, dtype=torch.long)

        if self.target_transform_fn is not None:
            y = self.target_transform_fn(y)

        return x, y

    def __len__(self) -> int:
        return len(self.data)
