#!/usr/bin/env python3
"""
Script to generate parquet files for QM9 molecule improvement training.

This script:
1. Loads QM9 dataset and computes fingerprints
2. Creates distance pairs using the filtered distance computation
3. Generates two parquet files:
   - molecules.parquet: SMILES, SELFIES, utility properties, and splits
   - pairs.parquet: item indices, distances, and splits
4. Ensures proper pair flipping for utility improvement
"""

# TODO: replace these with anderson split from atomic_datasets
# TODO: better qm9 cacheing of fingerprints, save to pkl 

import os
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

# QM9 and RDKit imports
from atomic_datasets import QM9
from atomic_datasets.utils.rdkit import is_molecule_sane
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import selfies as sf

# Import our distance functions
from distances import compute_filtered_pairs_blockwise_to_parquet, tanimoto_row_vs_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_valency_ok(mol):
    """Check if molecule has valid valency."""
    try:
        Chem.SanitizeMol(mol)
        return True
    except:
        return False


class QM9ParquetGenerator:
    """
    Generates parquet files for QM9 molecule improvement training.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        distance_threshold: float = 0.3,
        max_molecules: int = None,
        test_mode: bool = False,
        test_sizes: Tuple[int, int, int] = (500, 100, 100),  # (train, val, test)
        overwrite: bool = False
    ):
        """
        Initialize the QM9 parquet generator.
        
        Args:
            data_dir: Directory containing QM9 dataset
            output_dir: Directory to save parquet files
            distance_threshold: Distance threshold for filtering pairs
            max_molecules: Maximum number of molecules to process (None for all)
            test_mode: If True, creates small subsets for testing
            test_sizes: Tuple of (train_size, val_size, test_size) for test mode
            overwrite: Whether to overwrite existing files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.distance_threshold = distance_threshold
        self.max_molecules = max_molecules
        self.test_mode = test_mode
        self.test_sizes = test_sizes
        self.overwrite = overwrite
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.molecules_file = self.output_dir / "molecules.parquet"
        self.pairs_file = self.output_dir / "pairs.parquet"
        
        # Data storage
        self.dataset = None
        self.smiles_list = []
        self.selfies_list = []
        self.fingerprints = []
        self.properties = {}
        
    def load_qm9_dataset(self):
        """Load QM9 dataset and compute fingerprints."""
        logger.info(f"Loading QM9 dataset from {self.data_dir}")
        
        # Load dataset
        self.dataset = QM9(
            root_dir=str(self.data_dir),
            check_with_rdkit=True,
        )
        
        if self.max_molecules is not None:
            self.dataset = self.dataset[:self.max_molecules]
        
        logger.info(f"Loaded {len(self.dataset)} molecules from QM9")
        
        # Compute fingerprints and extract properties
        self._compute_fingerprints_and_properties()
        
    def _compute_fingerprints_and_properties(self):
        """Compute fingerprints and extract properties for all molecules."""
        logger.info("Computing fingerprints and extracting properties...")
        
        # Initialize fingerprint generator
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        
        # Initialize lists
        self.smiles_list = []
        self.selfies_list = []
        self.fingerprints = []
        
        # Initialize property dictionaries
        property_names = ['gap', 'homo', 'lumo', 'u0', 'u', 'h', 'g', 'cv', 'mu', 'alpha', 'r2', 'zpve']
        self.properties = {prop: [] for prop in property_names}
        
        num_errors = 0
        
        for i, graph in tqdm(enumerate(self.dataset), total=len(self.dataset), desc="Processing molecules"):
            mol = graph['properties']['rdkit_mol']
            
            # Compute SMILES
            smiles = Chem.MolToSmiles(mol)
            
            # Compute fingerprint
            fingerprint = mfpgen.GetFingerprint(mol)
            
            # Compute SELFIES (with error handling)
            try:
                selfies = sf.encoder(smiles)
            except Exception as e:
                val_ok = is_valency_ok(mol)
                is_sane = is_molecule_sane(mol)
                logger.warning(f'Error encoding SELFIES at index {i}, valency: {val_ok}, sanity: {is_sane}: {e}')
                num_errors += 1
                selfies = None
            
            # Extract properties
            props = graph['properties']
            for prop_name in property_names:
                value = props.get(prop_name, 0.0)
                # Handle NaN values
                if np.isnan(value):
                    value = 0.0
                self.properties[prop_name].append(float(value))
            
            # Store data
            self.smiles_list.append(smiles)
            self.selfies_list.append(selfies)
            self.fingerprints.append(fingerprint)
        
        logger.info(f"Processed {len(self.smiles_list)} molecules ({num_errors} SELFIES errors)")
        
    def create_train_val_test_splits(self, seed: int = 42):
        """Create train/val/test splits for molecules."""
        logger.info("Creating train/val/test splits...")
        
        n_molecules = len(self.smiles_list)
        indices = np.arange(n_molecules)
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        if self.test_mode:
            # Test mode: create small subsets
            train_size, val_size, test_size = self.test_sizes
            total_test_size = train_size + val_size + test_size
            
            if total_test_size > n_molecules:
                logger.warning(f"Requested test sizes ({total_test_size}) exceed available molecules ({n_molecules})")
                # Scale down proportionally
                scale_factor = n_molecules / total_test_size
                train_size = int(train_size * scale_factor)
                val_size = int(val_size * scale_factor)
                test_size = n_molecules - train_size - val_size
                logger.info(f"Scaled down to: train={train_size}, val={val_size}, test={test_size}")
            
            # Create small subsets
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:train_size + val_size + test_size]
            
            logger.info(f"Test mode splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        else:
            # Normal mode: use ratios
            train_end = int(n_molecules * 0.8) # Default to 80% for train
            val_end = train_end + int(n_molecules * 0.1) # Default to 10% for val
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
            
            logger.info(f"Normal mode splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # Create split assignments
        splits = ['train'] * n_molecules
        for idx in val_indices:
            splits[idx] = 'val'
        for idx in test_indices:
            splits[idx] = 'test'
        
        return splits
        
    def generate_molecules_parquet(self, splits: List[str]):
        """Generate the molecules parquet file."""
        logger.info("Generating molecules parquet file...")
        
        # Create molecules dataframe
        molecules_data = {
            'item_id': list(range(len(self.smiles_list))),
            'smiles': self.smiles_list,
            'selfies': self.selfies_list,
            'split': splits
        }
        
        # Add properties
        for prop_name, prop_values in self.properties.items():
            molecules_data[prop_name] = prop_values
        
        # Create dataframe and save
        molecules_df = pl.DataFrame(molecules_data)
        molecules_df.write_parquet(self.molecules_file)
        
        logger.info(f"Saved molecules to {self.molecules_file}")
        logger.info(f"Molecules dataframe shape: {molecules_df.shape}")
        logger.info(f"Columns: {molecules_df.columns}")
        
    def generate_pairs_parquet(self, splits: List[str]):
        """Generate the pairs parquet file using filtered distance computation."""
        logger.info("Generating pairs parquet file...")
        
        # Create temporary file for distance computation
        temp_pairs_file = self.output_dir / "temp_pairs.parquet"
        
        # Compute filtered distances
        logger.info(f"Computing distances with threshold {self.distance_threshold}")
        total_computed, total_stored = compute_filtered_pairs_blockwise_to_parquet(
            items=self.fingerprints,
            pairwise_func=tanimoto_row_vs_block,
            out_path=str(temp_pairs_file),
            distance_threshold=self.distance_threshold,
            smiles_list=self.smiles_list,  # For reference in output
            block_size=1000,
            write_every=1_000_000,
            max_blocks=-1,
            func_type='single_item',
            verbose=True
        )
        
        logger.info(f"Distance computation complete: {total_computed} computed, {total_stored} stored")
        
        # Load the computed pairs and add split information
        pairs_df = pl.read_parquet(temp_pairs_file)
        
        # Add split information based on the first molecule in each pair
        splits_array = np.array(splits)
        pair_splits = []
        
        for row in pairs_df.iter_rows(named=True):
            item_i = row['item_i']
            item_j = row['item_j']
            
            # Use the split of the first molecule (item_i)
            # This ensures pairs are assigned to the same split as their first molecule
            pair_splits.append(splits[item_i])
        
        # Add split column
        pairs_df = pairs_df.with_columns(pl.Series("split", pair_splits))
        
        # Save final pairs file
        pairs_df.write_parquet(self.pairs_file)
        
        # Remove temporary file
        temp_pairs_file.unlink()
        
        logger.info(f"Saved pairs to {self.pairs_file}")
        logger.info(f"Pairs dataframe shape: {pairs_df.shape}")
        logger.info(f"Columns: {pairs_df.columns}")
        
        # Print split statistics
        split_counts = pairs_df.group_by("split").count()
        logger.info("Pairs per split:")
        for row in split_counts.iter_rows(named=True):
            logger.info(f"  {row['split']}: {row['count']}")
        
    def generate_parquet_files(self, seed: int = 42):
        """Generate both parquet files."""
        if self.test_mode:
            logger.info(f"🧪 TEST MODE: Generating small subsets for testing")
            logger.info(f"   Train: {self.test_sizes[0]}, Val: {self.test_sizes[1]}, Test: {self.test_sizes[2]}")
        else:
            logger.info(f"📊 NORMAL MODE: Generating full dataset")
        
        logger.info(f"Generating parquet files in {self.output_dir}")
        
        # Check if files already exist
        if not self.overwrite and self.molecules_file.exists() and self.pairs_file.exists():
            logger.warning("Parquet files already exist. Use overwrite=True to regenerate.")
            return
        
        # Load dataset and compute fingerprints
        self.load_qm9_dataset()
        
        # Create splits
        splits = self.create_train_val_test_splits(seed)
        
        # Generate molecules parquet
        self.generate_molecules_parquet(splits)
        
        # Generate pairs parquet
        self.generate_pairs_parquet(splits)
        
        logger.info("Parquet file generation complete!")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Files created:")
        logger.info(f"  - {self.molecules_file}")
        logger.info(f"  - {self.pairs_file}")


def main():
    """Main function to generate QM9 parquet files."""
    
    # Configuration
    data_dir = os.getenv('DATA_DIR', '.')
    qm9_dir = os.path.join(data_dir, 'qm9')
    
    # Output configuration
    output_dir = os.path.join(data_dir, "cache/qm9_data_test2")
    distance_threshold = 0.3
    max_molecules = 10000  # Start with subset for testing
    
    # Test mode configuration
    test_mode = True  # Set to True for small subsets, False for full dataset
    test_sizes = (500, 100, 100)  # (train, val, test) sizes for testing
    
    # Create generator
    generator = QM9ParquetGenerator(
        data_dir=qm9_dir,
        output_dir=output_dir,
        distance_threshold=distance_threshold,
        max_molecules=max_molecules,
        test_mode=test_mode,
        test_sizes=test_sizes,
        overwrite=True  # Set to False to avoid regenerating
    )
    
    # Generate parquet files
    generator.generate_parquet_files(
        seed=42
    )
    
    print(f"\n✅ Parquet files generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🧪 Distance threshold: {distance_threshold}")
    print(f"🧪 Test mode: {test_mode}")
    if test_mode:
        print(f"🧪 Test sizes - Train: {test_sizes[0]}, Val: {test_sizes[1]}, Test: {test_sizes[2]}")
    print(f"🧪 Molecules processed: {max_molecules}")
    print(f"\nYou can now use these files with lobster training:")
    print(f"lobster_train data=molecule_improvement data.root='{output_dir}'")


if __name__ == "__main__":
    main()
