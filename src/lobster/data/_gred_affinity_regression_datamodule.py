from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from lobster.constants import Split
from lobster.datasets._gred_affinity_dataset import GredAffinityUMEStreamingDataset


class GredAffinityStreamingDataModule(LightningDataModule):
    """Lightning DataModule for GRED Affinity regression using UME streaming."""
    
    def __init__(
        self,
        target_column: str = "affinity_pkd",
        max_length: int = 2048,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
        significant_figures: int = 3,
        cache_dir: str | None = None,
        # Optional explicit validation split key (e.g., "val" or "test") if dataset lacks a canonical val split
        val_split_key: str | None = None,
    ):
        """
        Initialize the GRED Affinity streaming data module.
        
        Parameters
        ----------
        target_column : str, default="affinity_pkd"
            Target column for regression tasks. Options: "expression_yield", "affinity_pkd"
        max_length : int
            Maximum length for tokenization
        batch_size : int
            Batch size for data loaders
        num_workers : int
            Number of workers for data loading
        pin_memory : bool
            Whether to pin memory for faster GPU transfer
        seed : int
            Random seed for reproducibility
        significant_figures : int
            Number of significant figures for numerical values
        cache_dir : str | None
            Directory for caching dataset files
        """
        super().__init__()
        
        self.target_column = target_column
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.significant_figures = significant_figures
        self.cache_dir = cache_dir
        self.val_split_key = val_split_key
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets for each split."""
        
        if stage == "fit" or stage is None:
            self.train_dataset = GredAffinityUMEStreamingDataset(
                split=Split.TRAIN,
                target_column=self.target_column,
                significant_figures=self.significant_figures,
                seed=self.seed,
                cache_dir=self.cache_dir,
                max_length=self.max_length,
                val_split_key=self.val_split_key,
            )
            
            self.val_dataset = GredAffinityUMEStreamingDataset(
                split=Split.VALIDATION,
                target_column=self.target_column,
                significant_figures=self.significant_figures,
                seed=self.seed,
                cache_dir=self.cache_dir,
                max_length=self.max_length,
                val_split_key=self.val_split_key,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = GredAffinityUMEStreamingDataset(
                split=Split.TEST,
                target_column=self.target_column,
                significant_figures=self.significant_figures,
                seed=self.seed,
                cache_dir=self.cache_dir,
                max_length=self.max_length,
                val_split_key=self.val_split_key,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.train_dataset is None:
            raise ValueError("train_dataset is not initialized. Call setup() first.")
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.val_dataset is None:
            raise ValueError("val_dataset is not initialized. Call setup() first.")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.test_dataset is None:
            raise ValueError("test_dataset is not initialized. Call setup() first.")
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate function for affinity regression batches."""
        # Extract and stack the tokenized inputs
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        targets = []
        for item in batch:
            target_value = item.get("target")
            if target_value is None:
                raise ValueError(f"Missing target value in batch item. Available keys: {list(item.keys())}")
            targets.append(target_value)

        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets_tensor,
        }
