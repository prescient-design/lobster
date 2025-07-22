import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pooch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from lobster.constants import (
    CALM_TASK_SPECIES,
    CALMSpecies,
    CALMTask,
)

logger = logging.getLogger(__name__)


class CalmPropertyUnlabeledDataset(Dataset):
    """
    Unlabeled dataset aggregating all sequences from CALM tasks.

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
    include_tasks : List[CALMTask] | None, default=None
        Specific tasks to include (only used when computing from scratch).
    include_species : List[CALMSpecies] | None, default=None
        Specific species to include for species-specific tasks (only used when computing from scratch).
    cache_processed : bool, default=True
        Whether to cache the processed dataset.
    hf_repo_id : str | None, default="taylor-joren/calm-property-unlabeled"
        Hugging Face repository ID to load pre-computed data from.
        If None, will compute from scratch.
    force_recompute : bool, default=False
        If True, forces recomputation bypassing HF and cache.
    hf_config : Literal['train_only', 'full'] | None, default='train_only'
        Hugging Face configuration to load. Available options:
        - 'train_only': Train splits only (matches CalmLinearProbeCallback splits)
        - 'full': All sequences from all tasks
        If None, defaults to 'train_only'.
    random_seed : int, default=42
        Random seed for train/test splits when hf_config='train_only'.
        Should match the seed used in CalmLinearProbeCallback for consistency.
    test_size : float, default=0.2
        Fraction of data to use for test split when hf_config='train_only'.
    max_samples : int, default=3000
        Maximum number of samples per task when hf_config='train_only'.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform_fn: Callable | None = None,
        remove_duplicates: bool = True,
        min_length: int = 10,
        max_length: int | None = None,
        include_tasks: list[CALMTask] | None = None,
        include_species: list[CALMSpecies] | None = None,
        cache_processed: bool = True,
        hf_repo_id: str | None = "taylor-joren/calm-property-unlabeled",
        force_recompute: bool = False,
        hf_config: Literal["train_only", "full"] | None = "train_only",
        random_seed: int = 42,
        test_size: float = 0.2,
        max_samples: int = 3000,
    ):
        super().__init__()

        self.transform_fn = transform_fn
        self.remove_duplicates = remove_duplicates
        self.min_length = min_length
        self.max_length = max_length
        self.include_tasks = include_tasks or list(CALMTask)
        self.include_species = include_species
        self.cache_processed = cache_processed
        self.hf_repo_id = hf_repo_id
        self.force_recompute = force_recompute
        self.hf_config = hf_config or "train_only"
        self.random_seed = random_seed
        self.test_size = test_size
        self.max_samples = max_samples

        if root is None:
            root = pooch.os_cache("lobster")
        if isinstance(root, str):
            root = Path(root)
        self.root = root.resolve()

        # Create cache directory for processed data
        self.cache_dir = self.root / "calm_unlabeled"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load data - try HF first, then compute if needed
        self._load_data()

    def __getitem__(self, index: int) -> str:
        """Get a sequence by index."""
        sequence, _, _, _, _ = self.data.iloc[index]
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
            "tasks": dict(self.data["task"].value_counts()),
        }

    def get_data_source(self) -> str:
        """Get information about how the data was loaded."""
        return getattr(self, "_data_source", "unknown")

    def _compute_from_scratch(self) -> None:
        """Compute the dataset by loading and processing all CALM tasks."""
        all_sequences = []

        logger.info("Computing unlabeled dataset from CALM tasks...")

        # Process each task
        for task in tqdm(self.include_tasks, desc="Processing CALM tasks"):
            try:
                # Handle species-specific tasks
                if task in CALM_TASK_SPECIES:
                    # Process each species for species-specific tasks
                    species_list = self.include_species or CALM_TASK_SPECIES[task]
                    for species in species_list:
                        if species not in CALM_TASK_SPECIES[task]:
                            logger.warning(f"Species {species.value} not available for task {task.value}, skipping")
                            continue
                        try:
                            # Load the specific task/species combination
                            from lobster.datasets import CalmPropertyDataset

                            dataset = CalmPropertyDataset(task=task, species=species)

                            # Use the underlying DataFrame directly
                            df = dataset.data

                            if df.empty:
                                continue

                            # Apply train split if requested
                            if self.hf_config == "train_only":
                                df = self._apply_train_split(df)

                            # Extract sequences based on task type
                            task_sequences = self._extract_sequences(df, task, species.value)
                            all_sequences.extend(task_sequences)

                            logger.info(f"Extracted {len(task_sequences)} sequences from {task.value}/{species.value}")

                        except Exception as e:
                            logger.warning(f"Failed to process {task.value}/{species.value}: {e}")
                            continue
                else:
                    # Handle non-species-specific tasks
                    try:
                        from lobster.datasets import CalmPropertyDataset

                        dataset = CalmPropertyDataset(task=task)

                        # Use the underlying DataFrame directly
                        df = dataset.data

                        if df.empty:
                            continue

                        # Apply train split if requested
                        if self.hf_config == "train_only":
                            df = self._apply_train_split(df)

                        # Extract sequences based on task type
                        task_sequences = self._extract_sequences(df, task, None)
                        all_sequences.extend(task_sequences)

                        logger.info(f"Extracted {len(task_sequences)} sequences from {task.value}")

                    except Exception as e:
                        logger.warning(f"Failed to process {task.value}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Failed to process task {task.value}: {e}")
                continue

        if not all_sequences:
            raise ValueError("No sequences found in any of the specified tasks")

        # Convert to DataFrame
        df = pd.DataFrame(all_sequences, columns=["sequence", "sequence_type", "task", "species", "column"])

        # Apply filters
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
        if self.cache_processed:
            self._save_cached_data(df)

        self.data = df
        self._data_source = "computed"
        logger.info(f"Processed {len(self.data)} total sequences")

    def _apply_train_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply train split using same logic as CalmLinearProbeCallback."""
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        indices = np.arange(len(df))

        # If dataset is too large, subsample it first
        if len(indices) > self.max_samples:
            indices = np.random.choice(indices, size=self.max_samples, replace=False)

        # Create train/test split (we only want the train part)
        test_size = int(len(indices) * self.test_size)
        train_size = len(indices) - test_size
        shuffled_indices = np.random.permutation(indices)
        train_indices = shuffled_indices[:train_size]

        return df.iloc[train_indices]

    def _load_from_huggingface(self) -> pd.DataFrame | None:
        """Try to load dataset from Hugging Face."""
        if not self.hf_repo_id:
            return None

        try:
            logger.info(f"Loading from Hugging Face: {self.hf_repo_id}")

            # Load the dataset
            dataset = load_dataset(self.hf_repo_id, split="train")
            df = dataset.to_pandas()

            # Filter to the target configuration
            if "config_name" in df.columns:
                df = df[df["config_name"] == self.hf_config]
                df = df.drop(columns=["config_name"])

            logger.info(f"Loaded configuration '{self.hf_config}' with {len(df)} sequences")
            return df

        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            return None

    def _get_cache_filename(self) -> str:
        """Generate cache filename based on configuration."""
        return f"calm_unlabeled_{self.hf_config}.parquet"

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
        cache_file = self.cache_dir / self._get_cache_filename()
        logger.info(f"Saving processed data to {cache_file}")
        data.to_parquet(cache_file, index=False)

    def _extract_sequences(self, data: pd.DataFrame, task: CALMTask, species: str | None) -> list[tuple]:
        """Extract sequences from a dataset based on the task type."""
        sequences = []

        # Define the sequence columns for each task based on CalmPropertyDataset._set_columns
        sequence_columns = self._get_sequence_columns(task)

        for col in sequence_columns:
            # All CALM sequences are nucleotide sequences (coding DNA sequences)
            sequence_type = "nucleotide"

            if col in data.columns:
                col_sequences = data[col].dropna().tolist()
                # Store as tuples with (sequence, type, task, species, column)
                task_species = f"{task.value}_{species}" if species else task.value
                sequences.extend([(seq, sequence_type, task_species, species or "none", col) for seq in col_sequences])

        return sequences

    def _get_sequence_columns(self, task: CALMTask) -> list[str]:
        """Get the sequence columns for a given task based on CalmPropertyDataset logic."""
        # Based on the _set_columns method in CalmPropertyDataset
        if task in [CALMTask.FUNCTION_BP, CALMTask.FUNCTION_CC, CALMTask.FUNCTION_MF]:
            return ["sequence"]
        elif task == CALMTask.LOCALIZATION:
            return ["Sequence"]  # Note: capital S in localization task
        elif task == CALMTask.MELTOME:
            return ["sequence"]
        elif task in [CALMTask.SOLUBILITY, CALMTask.PROTEIN_ABUNDANCE, CALMTask.TRANSCRIPT_ABUNDANCE]:
            return ["cds"]
        else:
            # Default fallback
            return ["sequence", "cds", "Sequence"]
