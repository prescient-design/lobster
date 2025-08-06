import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
import pooch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from lobster.constants import (
    PEER_TASK_COLUMNS,
    PEER_TASK_SPLITS,
    PEERTask,
)

logger = logging.getLogger(__name__)


class PEERUnlabeledDataset(Dataset):
    """
    Unlabeled dataset aggregating all sequences from PEER tasks.

    By default, loads pre-computed data from Hugging Face for fast initialization.
    Falls back to computing from scratch if HF loading fails.

    Parameters
    ----------
    root : str | Path | None, default=None
        Root directory for data storage. If None, uses system cache.
    transform_fn : Optional[Callable], default=None
        Transformation to apply to sequences.
    remove_duplicates : bool, default=True
        Whether to remove duplicate sequences (only used when computing from scratch).
    min_length : int, default=10
        Minimum sequence length to include (only used when computing from scratch).
    max_length : int | None, default=None
        Maximum sequence length to include (only used when computing from scratch).
    include_tasks : List[PEERTask] | None, default=None
        Specific tasks to include (only used when computing from scratch).
    include_splits : List[str] | None, default=None
        Specific splits to include (only used when computing from scratch).
    cache_processed : bool, default=True
        Whether to cache the processed dataset.
    include_sequence_types : List[str] | None, default=None
        Types of sequences to include. Options: ['protein', 'ligand'].
        Determines which HF config to load.
    hf_repo_id : str | None, default="taylor-joren/peer-unlabeled"
        Hugging Face repository ID to load pre-computed data from.
        If None, will compute from scratch.
    force_recompute : bool, default=False
        If True, forces recomputation bypassing HF and cache.
    hf_config : Literal['train_only', 'full', 'protein_only', 'ligand_only', 'function_prediction', 'protein_ligand', 'medium_length'] | None, default='train_only'
        Hugging Face configuration to load. Available options:
        - 'train_only': All tasks, train split only (faster loading, for task-adaptive pre-training)
        - 'full': All tasks, all splits, all sequence types
        - 'protein_only': All tasks, all splits, protein sequences only
        - 'ligand_only': All tasks, all splits, ligand SMILES only
        - 'function_prediction': Function prediction tasks only
        - 'protein_ligand': Protein-ligand interaction tasks only
        - 'medium_length': All tasks, sequences 50-500 chars only
        If None, auto-selects based on include_sequence_types.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform_fn: Callable | None = None,
        remove_duplicates: bool = True,
        min_length: int = 10,
        max_length: int | None = None,
        include_tasks: list[PEERTask] | None = None,
        include_splits: list[str] | None = None,
        cache_processed: bool = True,
        include_sequence_types: list[str] | None = None,
        hf_repo_id: str | None = "taylor-joren/peer-unlabeled",
        force_recompute: bool = False,
        hf_config: Literal[
            "train_only",
            "full",
            "protein_only",
            "ligand_only",
            "function_prediction",
            "protein_ligand",
            "medium_length",
        ]
        | None = "train_only",
    ):
        super().__init__()

        self.transform_fn = transform_fn
        self.remove_duplicates = remove_duplicates
        self.min_length = min_length
        self.max_length = max_length
        self.include_tasks = include_tasks or list(PEERTask)
        self.include_splits = include_splits
        self.cache_processed = cache_processed
        self.include_sequence_types = include_sequence_types or ["protein", "ligand"]
        self.hf_repo_id = hf_repo_id
        self.force_recompute = force_recompute
        self.hf_config = hf_config

        if root is None:
            root = pooch.os_cache("lobster")
        if isinstance(root, str):
            root = Path(root)
        self.root = root.resolve()

        # Create cache directory for processed data
        self.cache_dir = self.root / "peer_unlabeled"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load data - try HF first, then compute if needed
        self._load_data()

    # === Main User-Facing Methods ===

    def __getitem__(self, index: int) -> str:
        """Get a sequence by index."""
        sequence, _, _, _ = self.data.iloc[index]
        if self.transform_fn is not None:
            sequence = self.transform_fn(sequence)
        return sequence

    def __len__(self) -> int:
        """Get total number of sequences."""
        return len(self.data)

    def _load_data(self) -> None:
        """Main data loading method - tries HF first, then cache, then computation."""
        self._data_source = "unknown"

        # Skip HF/cache if force recompute is set
        if self.force_recompute:
            logger.info("Force recompute enabled, computing from scratch")
            self._compute_from_scratch()
            return

        # Try loading from Hugging Face first
        data = self._load_from_huggingface()
        if data is not None:
            self.data = data
            self._data_source = f"huggingface:{self.hf_repo_id}"
            logger.info(f"Loaded {len(self.data)} sequences from Hugging Face")
            return

        # Try loading from cache
        data = self._load_cached_data()
        if data is not None:
            self.data = data
            self._data_source = "cache"
            logger.info(f"Loaded {len(self.data)} sequences from cache")
            return

        # Compute from scratch as last resort
        logger.info("Computing dataset from scratch...")
        self._compute_from_scratch()

    def get_sequence_stats(self) -> dict:
        """Get statistics about the sequences in the dataset."""
        lengths = self.data["sequence"].str.len()

        return {
            "total_sequences": len(self.data),
            "min_length": lengths.min(),
            "max_length": lengths.max(),
            "mean_length": lengths.mean(),
            "sequence_types": dict(self.data["sequence_type"].value_counts()),
            "tasks": dict(self.data["task"].value_counts()),
        }

    def get_length_distribution(self, bins: int = 50) -> pd.Series:
        """Get length distribution of sequences."""
        lengths = self.data["sequence"].str.len()
        return pd.cut(lengths, bins=bins).value_counts().sort_index()

    def get_sequences_by_length_range(self, min_len: int, max_len: int) -> list[str]:
        """Get sequences within a specific length range."""
        lengths = self.data["sequence"].str.len()
        mask = (lengths >= min_len) & (lengths <= max_len)
        return self.data[mask]["sequence"].tolist()

    def sample_sequences(self, n: int, random_state: int = 42) -> list[str]:
        """Sample n random sequences from the dataset."""
        return self.data.sample(n=n, random_state=random_state)["sequence"].tolist()

    def get_sequences_by_type(self, sequence_type: str) -> list[str]:
        """Get all sequences of a specific type (protein or ligand)."""
        return self.data[self.data["sequence_type"] == sequence_type]["sequence"].tolist()

    def get_sequence_type_counts(self) -> dict[str, int]:
        """Get counts of each sequence type."""
        return dict(self.data["sequence_type"].value_counts())

    def get_task_counts(self) -> dict[str, int]:
        """Get counts of sequences by task."""
        return dict(self.data["task"].value_counts())

    def get_data_source(self) -> str:
        """Get information about how the data was loaded."""
        return getattr(self, "_data_source", "unknown")

    @classmethod
    def list_hf_configurations(cls, repo_id: str = "taylor-joren/peer-unlabeled") -> dict[str, dict]:
        """List available configurations in the Hugging Face dataset."""
        try:
            dataset = load_dataset(repo_id, split="train")
            df = dataset.to_pandas()

            # Group by configuration and get stats for each
            configs = {}
            for config_name in df["config_name"].unique():
                config_data = df[df["config_name"] == config_name]
                configs[config_name] = {
                    "num_sequences": len(config_data),
                    "sequence_types": dict(config_data["sequence_type"].value_counts()),
                    "tasks": dict(config_data["task"].value_counts()),
                }

            return configs
        except Exception as e:
            logger.error(f"Failed to list HF configurations: {e}")
            return {}

    # === Forcing recomputation ===

    def _compute_from_scratch(self) -> None:
        """Compute the dataset by loading and processing all PEER tasks."""
        all_sequences = []

        logger.info("Computing unlabeled dataset from PEER tasks...")

        # Process each task
        for task in tqdm(self.include_tasks, desc="Processing PEER tasks"):
            if task not in PEER_TASK_SPLITS:
                logger.warning(f"No splits defined for task {task}, skipping")
                continue

            # Process each split for this task
            splits_to_process = self.include_splits or PEER_TASK_SPLITS[task]
            for split in splits_to_process:
                if split not in PEER_TASK_SPLITS[task]:
                    logger.warning(f"Split {split} not available for task {task}, skipping")
                    continue

                try:
                    # Load the specific task/split
                    from lobster.datasets import PEERDataset

                    dataset = PEERDataset(task=task, split=split)

                    # Use the underlying DataFrame directly
                    df = dataset.data

                    if df.empty:
                        continue

                    # Extract sequences based on task type
                    task_sequences = self._extract_sequences(df, task)
                    all_sequences.extend(task_sequences)

                    logger.info(f"Extracted {len(task_sequences)} sequences from {task.value}/{split}")

                except Exception as e:
                    logger.warning(f"Failed to process {task.value}/{split}: {e}")
                    continue

        if not all_sequences:
            raise ValueError("No sequences found in any of the specified tasks/splits")

        # Convert to DataFrame
        df = pd.DataFrame(all_sequences, columns=["sequence", "sequence_type", "task", "column"])

        # Apply filters
        # Length filtering
        if self.min_length is not None:
            df = df[df["sequence"].str.len() >= self.min_length]
        if self.max_length is not None:
            df = df[df["sequence"].str.len() <= self.max_length]

        # Remove duplicates if requested
        if self.remove_duplicates:
            initial_count = len(df)
            df = df.drop_duplicates(subset=["sequence"])
            logger.info(f"Removed {initial_count - len(df)} duplicate sequences")

        # Cache the processed data
        self._save_cached_data(df)

        self.data = df
        self._data_source = "computed"
        logger.info(f"Processed {len(self.data)} total sequences")

    def _load_from_huggingface(self) -> pd.DataFrame | None:
        """Try to load dataset from Hugging Face."""
        if not self.hf_repo_id:
            return None

        try:
            logger.info(f"Loading from Hugging Face: {self.hf_repo_id}")

            # Load the dataset
            dataset = load_dataset(self.hf_repo_id, split="train")
            df = dataset.to_pandas()

            # Determine which config to use
            target_config = self._determine_hf_config()
            available_configs = df["config_name"].unique()

            if target_config not in available_configs:
                logger.warning(f"Configuration '{target_config}' not found, using 'full'")
                target_config = "full"

                if target_config not in available_configs:
                    logger.warning("No suitable configuration found in HF dataset")
                    return None

            # Filter to the target configuration
            df = df[df["config_name"] == target_config]

            # Drop the config_name column since we don't need it anymore
            df = df.drop(columns=["config_name"])

            logger.info(f"Loaded configuration '{target_config}' with {len(df)} sequences")
            return df

        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            return None

    def _determine_hf_config(self) -> str:
        """Determine which HF config to load based on current parameters."""
        # Use explicit config if provided
        if self.hf_config:
            return self.hf_config

        # Match based on sequence types (fallback when hf_config is None)
        if self.include_sequence_types == ["protein"]:
            return "protein_only"
        elif self.include_sequence_types == ["ligand"]:
            return "ligand_only"

        # Default fallback
        return "train_only"

    def _get_cache_filename(self) -> str:
        """Generate cache filename based on configuration."""
        # Simplified cache filename - most users will use default settings
        types_str = "_".join(sorted(self.include_sequence_types))
        return f"peer_unlabeled_{types_str}.parquet"

    def _load_cached_data(self) -> pd.DataFrame | None:
        """Try to load cached processed data."""
        if not self.cache_processed:
            return None

        cache_file = self.cache_dir / self._get_cache_filename()
        if cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)
        return None

    def _save_cached_data(self, data: pd.DataFrame) -> None:
        """Save processed data to cache."""
        if not self.cache_processed:
            return

        cache_file = self.cache_dir / self._get_cache_filename()
        logger.info(f"Saving processed data to {cache_file}")
        data.to_parquet(cache_file, index=False)

    def _extract_sequences(self, data: pd.DataFrame, task: PEERTask) -> list[tuple]:
        """Extract sequences from a dataset based on the task type and included sequence types."""
        sequences = []

        if task not in PEER_TASK_COLUMNS:
            logger.debug(f"Task {task} not in PEER_TASK_COLUMNS")
            return sequences

        input_cols, _ = PEER_TASK_COLUMNS[task]
        logger.debug(f"Task {task}: input_cols={input_cols}, available_cols={list(data.columns)}")

        for col in input_cols:
            sequence_type = None

            # Determine sequence type
            if "protein" in col.lower() and "sequence" in col.lower():
                sequence_type = "protein"
            elif "ligand" in col.lower() and "smiles" in col.lower():
                sequence_type = "ligand"

            logger.debug(
                f"Column {col}: sequence_type={sequence_type}, in_include_types={sequence_type in self.include_sequence_types if sequence_type else False}"
            )

            # Only include if the sequence type is requested
            if sequence_type and sequence_type in self.include_sequence_types:
                if col in data.columns:
                    col_sequences = data[col].dropna().tolist()
                    logger.debug(f"Column {col}: found {len(col_sequences)} sequences")
                    # Store as tuples with (sequence, type, task, column)
                    sequences.extend([(seq, sequence_type, task.value, col) for seq in col_sequences])
                else:
                    logger.debug(f"Column {col} not found in data")

        logger.debug(f"Task {task}: extracted {len(sequences)} total sequences")
        return sequences
