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
        pair_filename: str = "pairs.parquet", # name of the pairs file within root
        utility_key: str = "utility_0",  # Which utility column to use
        delta: float = None,  # Distance threshold
        epsilon: float = 0.1,  # Utility improvement threshold
        transform_fn: Optional[Callable] = None,  # Tokenizer transform function
        distance_type: Optional[str] = None,  # "morgan_tanimoto_distance" or "shape_tanimoto_distance"
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
        
        # Load pairs table
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
            if abs(utility_improvement) > self.epsilon:
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
        tokenized = self.transform(sequence)

        # seq_len = tokenized["input_ids"].shape[1]
        # causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        res = {
            "input_ids": tokenized["input_ids"], #[:-1],
            "labels": tokenized["input_ids"], #[1:],
            "attention_mask": tokenized["attention_mask"],
        }

        # print('attention_mask from tokenized', tokenized['attention_mask'].shape)
        # print('attention_mask from res', res['attention_mask'].shape)
        # raise Exception("stop")
        
        # Add utility improvement as additional metadata
        # tokenized["utility_improvement"] = torch.tensor([utility_improvement], dtype=torch.float32)
        
        return res, [] # as label


class MoleculeImprovementLightningDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for molecule improvement training.
    
    This follows lobster's conventions and integrates with the existing training pipeline.
    """
    
    def __init__(
        self,
        root: str,  # Data root directory
        pair_filename: str = "pairs.parquet", # name of the pairs file within root
        utility_key: str = "utility_0",  # Which utility column to use
        delta: float = 0.3,
        epsilon: float = 0.1,
        transform_fn: Optional[Callable] = None,  # Tokenizer transform function
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
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
        """
        super().__init__()

        self._root = root
        self._pair_filename = pair_filename
        self._utility_key = utility_key
        self._delta = delta
        self._epsilon = epsilon
        self._transform_fn = transform_fn
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        
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
                pair_filename=self._pair_filename,
                split="train",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
            )
            
            self._val_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._pair_filename,
                split="val",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
            )
            
            self._test_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._pair_filename,
                split="test",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
            )
            
        if stage == "predict":
            # For prediction, use the test dataset
            self._test_dataset = MoleculeImprovementDataset(
                root=self._root,
                pair_filename=self._pair_filename,
                split="test",
                utility_key=self._utility_key,
                delta=self._delta,
                epsilon=self._epsilon,
                transform_fn=self._transform_fn,
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