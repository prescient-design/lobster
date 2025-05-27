import ast
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pooch
import torch
from datasets import load_dataset
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
    transform_fn : Optional[Callable], default=None
        Transformation to apply to input sequences.
    target_transform_fn : Optional[Callable], default=None
        Transformation to apply to target values.
    columns : Optional[Sequence[str]], default=None
        Specific columns to use from the dataset. If None, appropriate columns are
        selected based on the task.
    """

    def __init__(
        self,
        task: PEERTask | str,
        root: str | Path | None = None,
        *,
        split: str = "train",
        transform_fn: Callable | None = None,
        target_transform_fn: Callable | None = None,
        columns: Sequence[str] | None = None,
    ):
        super().__init__()

        # Convert string task to enum if needed
        if isinstance(task, str):
            task = PEERTask(task)

        self.task = task
        self.split = split
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn

        # Validate the split is available for this task
        if split not in PEER_TASK_SPLITS[task]:
            raise ValueError(f"Invalid split '{split}' for task '{task}'. Available splits: {PEER_TASK_SPLITS[task]}")

        self.task_type, self.num_classes = PEER_TASKS[task]
        self.task_category = PEER_TASK_CATEGORIES[task]

        if root is None:
            root = pooch.os_cache("lbster")
        if isinstance(root, str):
            root = Path(root)
        self.root = root.resolve()

        # Configure paths and load data
        self.hf_data_file = self._configure_hf_path()
        self._load_data()
        self._set_columns(columns)

    def _configure_hf_path(self) -> str:
        """Get Hugging Face data path based on task and split."""
        task_category = self.task_category.value
        task = self.task.value
        split = self.split

        # Path for Hugging Face repo
        return f"{task_category}/{task}/{split}.parquet"

    def _load_data(self) -> None:
        """Load data from Hugging Face."""
        dataset = load_dataset("taylor-joren/peer", data_files=self.hf_data_file, split="train")
        self.data = dataset.to_pandas()

    def _set_columns(self, columns=None):
        """Set the columns to use based on the task"""
        if columns is None:
            if self.task not in PEER_TASK_COLUMNS:
                raise ValueError(f"No column mapping defined for task '{self.task}'")

            input_cols, target_cols = PEER_TASK_COLUMNS[self.task]
            columns = input_cols + target_cols

            for col in columns:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Expected column '{col}' not found in dataset for task '{self.task}'. "
                        f"Available columns: {list(self.data.columns)}"
                    )

        self.columns = columns

    @staticmethod
    def string_to_tensor(
        string_data: str, dtype: torch.dtype = torch.float32, shape: tuple | None = None
    ) -> torch.Tensor:
        """
        Convert string representation directly to PyTorch tensor.
        Handles various input formats including tensors, arrays, lists, and string representations.
        """
        # Handle cases where input is already a tensor or array
        if isinstance(string_data, torch.Tensor):
            return string_data.to(dtype=dtype)

        if isinstance(string_data, np.ndarray):
            return torch.tensor(string_data, dtype=dtype)

        if isinstance(string_data, list):
            return torch.tensor(string_data, dtype=dtype)

        # Parse string representations
        try:
            # Try parsing with ast.literal_eval for safety
            parsed_data = ast.literal_eval(string_data)
            tensor = torch.tensor(parsed_data, dtype=dtype)
        except (ValueError, SyntaxError, TypeError):
            # Fallback to regex extraction for non-standard formats
            numeric_values = re.findall(r"-?\d+\.\d+|-?\d+", string_data)
            tensor = torch.tensor([float(x) for x in numeric_values], dtype=dtype)

        # Reshape if needed and possible
        if shape and tensor.numel() == np.prod(shape):
            tensor = tensor.reshape(shape)
        elif shape and shape[0] == -1 and tensor.numel() % shape[1] == 0:
            # For cases like (-1, 3) where we want to reshape to nx3 matrix
            tensor = tensor.reshape(-1, shape[1])

        return tensor

    def __getitem__(self, index: int):
        """
        Get a single item from the dataset with task-specific handling.
        Returns a tuple of (input, target) where format depends on the specific task.
        """
        item = self.data.iloc[index]

        # Extract input features based on task
        if self.task in PEER_TASK_COLUMNS:
            input_cols, target_cols = PEER_TASK_COLUMNS[self.task]
            if len(input_cols) == 1:
                x = item[input_cols[0]]
                if self.transform_fn is not None:
                    x = self.transform_fn(x)
            else:
                x = [item[col] for col in input_cols]
                if self.transform_fn is not None:
                    x = [self.transform_fn(xi) for xi in x]
        else:
            x = item[self.columns[0]]
            if self.transform_fn is not None:
                x = self.transform_fn(x)

        # Handle target data
        if self.task == PEERTask.PROTEINNET:
            # Special handling for contact map prediction
            tertiary_data = PEERDataset.string_to_tensor(item["tertiary"])
            valid_mask = PEERDataset.string_to_tensor(item["valid_mask"])
            y = (tertiary_data, valid_mask)

        elif self.task == PEERTask.SECONDARY_STRUCTURE:
            y = torch.tensor(item["ss3"], dtype=torch.long)

        else:
            # Generic handling based on task type
            if self.task_type == "regression":
                if isinstance(item[target_cols[0]], torch.Tensor):
                    y = item[target_cols[0]]
                    # Special handling for tasks that need first element extraction
                    if self.task in [PEERTask.FLUORESCENCE, PEERTask.STABILITY] and y.numel() > 1:
                        y = y[0].reshape(1)
                else:
                    y = torch.tensor(item[target_cols[0]], dtype=torch.float32)
            else:
                # Classification tasks (binary & multiclass)
                if isinstance(item[target_cols[0]], torch.Tensor):
                    y = item[target_cols[0]]
                else:
                    y = torch.tensor(item[target_cols[0]], dtype=torch.long)

        if self.target_transform_fn is not None:
            y = self.target_transform_fn(y)

        return x, y

    def __len__(self) -> int:
        return len(self.data)
