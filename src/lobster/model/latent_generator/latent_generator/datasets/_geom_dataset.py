import pandas as pd
import boto3
import s3fs
from pathlib import Path
import torch
from loguru import logger
from typing import Dict, Any, Optional
import numpy as np
import os
import pathlib
from typing import Callable, Optional, Union
import torch
from torch_geometric.data import Dataset
from loguru import logger
from icecream import ic

def load_geom_parquet_from_s3(s3_path: str = "s3://prescient-lobster/ume/datasets/geom/processed/test/partition_0000.parquet") -> pd.DataFrame:
    """
    Load a parquet file from S3 containing GEOM dataset data.
    
    Args:
        s3_path: S3 path to the parquet file
        
    Returns:
        pandas DataFrame containing the data
    """
    try:
        logger.info(f"Loading parquet file from: {s3_path}")
        
        # Use s3fs to read parquet file directly
        fs = s3fs.S3FileSystem()
        df = pd.read_parquet(s3_path, filesystem=fs)
        
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading parquet file: {e}")
        raise

def examine_geom_data(df: pd.DataFrame) -> None:
    """
    Examine the structure and content of the GEOM dataset.
    
    Args:
        df: DataFrame containing GEOM data
    """
    logger.info("=== GEOM Dataset Examination ===")
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Show data types
    logger.info("\nData types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    # Show first few rows
    logger.info("\nFirst 3 rows:")
    logger.info(df.head(3))
    
    # Show basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info(f"\nNumeric columns statistics:")
        logger.info(df[numeric_cols].describe())
    
    # Show unique values for categorical columns (first few)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        logger.info(f"\nCategorical columns sample values:")
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            unique_vals = df[col].unique()
            logger.info(f"  {col}: {len(unique_vals)} unique values")
            if len(unique_vals) <= 10:
                logger.info(f"    Values: {unique_vals}")
            else:
                logger.info(f"    Sample values: {unique_vals[:5]}...")

class GeomLigandDataset(Dataset):
    """Dataset class for ligand atom coordinates.
    Expects .pt files with a 'coords' key for atom coordinates.
    """
    def __init__(
        self,
        root: Union[str, os.PathLike]="s3://prescient-lobster/ume/datasets/geom/processed/test",
        transform_protein: Optional[Callable] = None,
        transform_ligand: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        min_len: int = 1,
        testing: bool = False,
    ):
        self.root = str(root)  # Keep as string for S3 compatibility
        self.transform_ligand = transform_ligand
        self.pre_transform = pre_transform
        self.min_len = min_len
        self.testing = testing
        self._load_data()
        logger.info(f"Loaded ligand data points.")
        super().__init__(root, transform_protein, transform_ligand, pre_transform)

    def _load_data(self):
        """Load data files from S3 or local filesystem."""
        processed_files_ligand = []
        
        if self.root.startswith("s3://"):
            # Handle S3 paths
            fs = s3fs.S3FileSystem()
            try:
                # List all files in the S3 directory recursively
                # Use find() instead of ls() for recursive listing
                all_files = fs.find(self.root)
                
                # Filter for ligand.parquet files
                for file_path in all_files:
                    if file_path.endswith(".parquet"):
                        processed_files_ligand.append(f"s3://{file_path}")
                        
                logger.info(f"Found {len(processed_files_ligand)} ligand files in S3")
                
            except Exception as e:
                logger.error(f"Error listing S3 files: {e}")
                raise
        else:
            # Handle local filesystem paths
            for root, dirs, files in os.walk(self.root):
                for file in files:
                    if file.endswith(".parquet"):
                        processed_files_ligand.append(os.path.join(root, file))

        self.dataset_filenames_ligand = processed_files_ligand
        self.dataset_filenames_ligand.sort()

        logger.info(f"Loaded {len(self.dataset_filenames_ligand)} ligand data points.")

        #make tuple of ligand and protein if pdb_id is the same
        self.dataset_filenames = []
        
        for ligand_file in self.dataset_filenames_ligand:
            self.dataset_filenames.append(ligand_file)

    def len(self) -> int:
        return len(self.dataset_filenames)

    def __getitem__(self, idx: int):
        file_path = self.dataset_filenames[idx]
        
        # Handle S3 vs local file loading
        if file_path.startswith("s3://"):
            # Load from S3
            fs = s3fs.S3FileSystem()
            with fs.open(file_path, 'rb') as f:
                x_ligand = pd.read_parquet(f)
        else:
            # Load from local filesystem
            x_ligand = pd.read_parquet(file_path)
        
        #randomly pick a row
        x_ligand = x_ligand.sample(1)
        ic(x_ligand)
        x_ligand = x_ligand.to_dict(orient="records")[0]
        ic(x_ligand)

        # x should have 'coords' key: [num_atoms, 3]
        if self.transform_ligand:
            x_ligand = self.transform_ligand(x_ligand)

        return {"protein": None, "ligand": x_ligand}

if __name__ == "__main__":
    # Load the data
    #df = load_geom_parquet_from_s3()
    
    # Examine the data
    #examine_geom_data(df)
    
    # Test the GeomLigandDataset with S3
    logger.info("\n=== Testing GeomLigandDataset with S3 ===")
    try:
        dataset = GeomLigandDataset(root="s3://prescient-lobster/ume/datasets/geom/processed/test")
        logger.info(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            # Test loading first item
            first_item = dataset[0]
            logger.info(f"First item keys: {list(first_item.keys())}")
            if 'ligand' in first_item and first_item['ligand'] is not None:
                logger.info(f"Ligand data type: {type(first_item['ligand'])}")
                if hasattr(first_item['ligand'], 'keys'):
                    logger.info(f"Ligand keys: {list(first_item['ligand'].keys())}")
                elif hasattr(first_item['ligand'], 'shape'):
                    logger.info(f"Ligand shape: {first_item['ligand'].shape}")
        
    except Exception as e:
        logger.error(f"Error testing GeomLigandDataset: {e}")
        import traceback
        traceback.print_exc()
