import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split

from lobster.tokenization import SmilesTokenizerFast
from lobster.transforms import TokenizerTransform
import pdb
import pandas as pd
from qm9_pair_gen.utils_mol import filter_top_pairs_per_molecule



class MoleculeImprovementDataset(Dataset):
    """
    Dataset for training a model to improve molecules based on cached distance pairs.
    
    Loads two tables from a root directory:
    1. molecules.parquet: SMILES, multiple utility scores, and splits
    2. pairs.parquet: item indices, distances, and splits
    
    Filters by distance threshold, utility improvement, and split.
    Ensures the second molecule in each pair is always the better one.
    Training format: <bos>x<sep>x'<eos> where x' is the improved molecule
    """
    
    def __init__(
        self,
        root: str,  # Data root directory containing molecules.parquet and pairs.parquet
        split: str,  # "train", "val", or "test"
        pair_filename: str = "pairs.parquet", # name of the pairs file within root, contains shape and regular tanimoto distances
        utility_key: str = "utility_0",  # Which utility column to use
        delta: float = None,  # Distance threshold
        epsilon: float = 0.1,  # Utility improvement threshold
        transform_fn: Optional[Callable] = None,  # Tokenizer transform function
        distance_type: Optional[str] = None,  # "morgan_tanimoto_distance" or "shape_tanimoto_distance"
        shape_tanimoto_percentile: Optional[float] = None,  # Keep closest X% of pairs by shape Tanimoto distance per molecule
        shape_tanimoto_num_pairs: Optional[int] = None,  # Keep closest N pairs by shape Tanimoto distance per molecule
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory containing molecules.parquet and pairs.parquet
            split: Which split to use ("train", "val", "test")
            utility_key: Which utility column to use (e.g., "utility_0", "gap", "homo", etc.)
            delta: Maximum distance threshold for molecule pairs
            epsilon: Minimum utility improvement threshold
            transform_fn: Tokenizer transform function (will create default if None)
        """
        self.root = Path(root)
        self.split = split
        self.utility_key = utility_key
        self.delta = delta
        self.epsilon = epsilon
        self.distance_type = distance_type
        self.shape_tanimoto_percentile = shape_tanimoto_percentile
        self.shape_tanimoto_num_pairs = shape_tanimoto_num_pairs
        
        # Set up file paths using standardized naming
        self.molecules_parquet_path = self.root / "molecules.parquet"
        self.pairs_parquet_path = self.root / pair_filename
        
        # Check if files exist
        if not self.molecules_parquet_path.exists():
            raise FileNotFoundError(f"Molecules file not found: {self.molecules_parquet_path}")
        if not self.pairs_parquet_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {self.pairs_parquet_path}")
        
        # Initialize transform
        if transform_fn is None:
            # maybe get rid of this? shouldn't be called
            # Create default SMILES tokenizer transform
            from lobster.tokenization import SmilesTokenizerFast
            from lobster.transforms import TokenizerTransform
            
            tokenizer = SmilesTokenizerFast()
            self.transform = TokenizerTransform(
                tokenizer,
                padding="max_length",
                truncation=True,
                max_length=512,  # Default max length
            )
        else:
            self.transform = transform_fn
        
        # Load and filter training pairs
        self.training_pairs = self._load_and_filter_pairs()
        
    def _load_and_filter_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Load data from both tables and filter by distance, utility, and split.
        Ensures the second molecule is always the better one.
        """
        import polars as pl
        
        # Load molecules table
        molecules_df = pl.read_parquet(self.molecules_parquet_path)

        # I think this is unnecessary
        if "split" in molecules_df.columns:
            molecules_df = molecules_df.filter(pl.col("split") == self.split)
        else:
            molecules_df = molecules_df.filter(pl.col(self.split) == True)
        
        # Check if utility column exists
        if self.utility_key not in molecules_df.columns:
            available_columns = [col for col in molecules_df.columns if col.startswith('utility_') or col in ['gap', 'homo', 'lumo', 'u0', 'u', 'h', 'g', 'cv', 'mu', 'alpha', 'r2', 'zpve']]
            raise ValueError(f"Utility column '{self.utility_key}' not found. Available utility columns: {available_columns}")
        
        # Create lookup dictionaries
        smiles_lookup = {row['item_id']: row['smiles'] for row in molecules_df.iter_rows(named=True)}
        utility_lookup = {row['item_id']: row[self.utility_key] for row in molecules_df.iter_rows(named=True)}
        
        # Load pairs table. not using scan_parquet, so it's loading everything I think?
        pairs_df = pl.read_parquet(self.pairs_parquet_path)
        # removing because pairs should only be created from train split already, by qm9_pair_gen script
        #pairs_df = pairs_df.filter(pl.col("split") == self.split)
        
        print(f"Loaded {len(pairs_df)} pairs for split '{self.split}' using utility '{self.utility_key}'")
        
        # Filter by distance threshold if specified
        if self.delta is not None and self.delta != 'None':
            if self.distance_type is None:
                raise ValueError(f"Distance type must be specified if delta is provided as {self.delta}")
            pairs_df = pairs_df.filter(pl.col(self.distance_type) < self.delta)
            print(f"After distance filtering (< {self.delta}): {len(pairs_df)} pairs")
        
        # Filter by shape Tanimoto if specified
        if self.shape_tanimoto_percentile is not None or self.shape_tanimoto_num_pairs is not None:
            # Convert to pandas DataFrame for the utility function
            
            if self.shape_tanimoto_num_pairs is not None and self.shape_tanimoto_percentile is not None:
                pairs_pandas = pairs_df.to_pandas()
                
                # Use the utility function to filter by shape Tanimoto
                filtered_pairs_pandas = filter_top_pairs_per_molecule(
                    pairs_pandas, 
                    property_key='shape_tanimoto_distance',
                    percentile=self.shape_tanimoto_percentile,
                    num_pairs=self.shape_tanimoto_num_pairs
                )
            
                # Convert back to polars DataFrame
                pairs_df = pl.from_pandas(filtered_pairs_pandas)
            
            if self.shape_tanimoto_percentile is not None:
                print(f"After shape Tanimoto percentile filtering ({self.shape_tanimoto_percentile}%): {len(pairs_df)} pairs")
            else:
                print(f"After shape Tanimoto num_pairs filtering ({self.shape_tanimoto_num_pairs} pairs per molecule): {len(pairs_df)} pairs")
        
        # Filter by utility improvement and ensure correct ordering
        training_pairs = []
        for row in pairs_df.iter_rows(named=True):
            item_i = row['molecule_a_idx']
            item_j = row['molecule_b_idx']
            #distance = row['distance']
            
            # Get SMILES and utilities
            smiles_i = smiles_lookup.get(item_i)
            smiles_j = smiles_lookup.get(item_j)
            utility_i = utility_lookup.get(item_i)
            utility_j = utility_lookup.get(item_j)
            
            # Skip if we don't have data for these items
            if smiles_i is None or smiles_j is None or utility_i is None or utility_j is None:
                continue
            
            # Calculate utility improvement
            utility_improvement = utility_j - utility_i
            
            # Check if improvement meets threshold
            if abs(utility_improvement) < self.epsilon:
                #print(f"Skipping pair {smiles_i} {smiles_j} because utility improvement {utility_improvement} is less than {self.epsilon}")
                continue
                
            # Ensure the second molecule is always the better one
            if utility_improvement > 0:
                # Original order is correct: (worse, better)
                training_pairs.append((smiles_i, smiles_j, utility_improvement))
            else:
                # Flip order: (worse, better)
                training_pairs.append((smiles_j, smiles_i, -utility_improvement))
        
        print(f"After utility filtering (> {self.epsilon}): {len(training_pairs)} pairs")
        return training_pairs
    
    def __len__(self) -> int:
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        smiles1, smiles2, utility_improvement = self.training_pairs[idx]
        
        # Construct the training sequence: <bos>smiles1<sep>smiles2<eos>
        # smiles1 is the worse molecule, smiles2 is the better molecule
        sequence = f"{smiles1}<sep>{smiles2}"
        
        # Tokenize the sequence using the transform function
        tokenized = self.transform(sequence) #[smiles1, smiles2]) #sequence)

        # Create labels tensor and mask out padding tokens with -100
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100 # new based on docs https://huggingface.co/docs/transformers/en/model_doc/llama#transformers.LlamaForCausalLM

        res = {
            "input_ids": tokenized["input_ids"],
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
        }

        return res #, [] # as label # removing this


class MoleculeImprovementLightningDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for molecule improvement training.
    
    This follows lobster's conventions and integrates with the existing training pipeline.
    """
    
    def __init__(
        self,
        root: str,  # Data root directory
        train_pair_filename: str = "pairs_train.parquet", # name of the pairs file within root
        val_pair_filename: str = "pairs_val.parquet", # name of the pairs file within root
        test_pair_filename: str = "pairs_test.parquet", # name of the pairs file within root
        utility_key: str = "utility_0",  # Which utility column to use
        delta: float = 0.3,
        epsilon: float = 0.1,
        transform_fn: Optional[Callable] = None,  # Tokenizer transform function
        shape_tanimoto_percentile: Optional[float] = None,  # Keep closest X% of pairs by shape Tanimoto distance per molecule
        shape_tanimoto_num_pairs: Optional[int] = None,  # Keep closest N pairs by shape Tanimoto distance per molecule
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        max_train_samples: Optional[int] = None,
    ) -> None:
        """
        Initialize the MoleculeImprovementLightningDataModule.
        
        Args:
            root: Root directory containing molecules.parquet and pairs.parquet
            utility_key: Which utility column to use (e.g., "utility_0", "gap", "homo", etc.)
            delta: Distance threshold for molecule pairs
            epsilon: Utility improvement threshold
            transform_fn: Tokenizer transform function
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop the last incomplete batch
            max_train_samples: Maximum number of training samples to use (for debugging)
        """
        super().__init__()

        # Validate shape Tanimoto filtering parameters
        if shape_tanimoto_percentile is not None and shape_tanimoto_num_pairs is not None:
            raise ValueError("Cannot specify both shape_tanimoto_percentile and shape_tanimoto_num_pairs. Use one or the other.")

        self._root = root
        self._train_pair_filename = train_pair_filename
        self._val_pair_filename = val_pair_filename
        self._test_pair_filename = test_pair_filename
        self._utility_key = utility_key
        self._delta = delta
        self._epsilon = epsilon
        self._transform_fn = transform_fn
        self._shape_tanimoto_percentile = shape_tanimoto_percentile
        self._shape_tanimoto_num_pairs = shape_tanimoto_num_pairs
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._max_train_samples = max_train_samples
        
        # Initialize datasets
        self._train_dataset: Union[MoleculeImprovementDataset, None] = None
        self._val_dataset: Union[MoleculeImprovementDataset, None] = None
        self._test_dataset: Union[MoleculeImprovementDataset, None] = None
        
    def prepare_data(self) -> None:
        """Prepare the dataset by creating training pairs."""
        # This is called once per node
        pass
        
    def setup(self, stage: str = "fit") -> None:
        """Set up datasets for training, validation, and testing."""
        if stage == "fit" or stage == "test":
            # Create datasets for each split
            self._train_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._train_pair_filename,
                split="train",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
                shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
            )

            if self._max_train_samples is not None:
                self._train_dataset, _ = random_split(self._train_dataset, [self._max_train_samples, len(self._train_dataset) - self._max_train_samples])
            
            self._val_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._val_pair_filename,
                split="val",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
                shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
            )
            
            self._test_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._test_pair_filename,
                split="test",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
                shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
            )
            
        if stage == "predict":
            # For prediction, use the test dataset
            self._test_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._test_pair_filename,
                split="test",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
                shape_tanimoto_percentile=self._shape_tanimoto_percentile,
                shape_tanimoto_num_pairs=self._shape_tanimoto_num_pairs,
            )
            
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self._train_dataset is None:
            raise ValueError("Train dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=self._drop_last,
        )
        
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self._val_dataset is None:
            raise ValueError("Validation dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        )
        
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self._test_dataset is None:
            raise ValueError("Test dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        )
        
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader."""
        if self._test_dataset is None:
            raise ValueError("Test dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        ) 