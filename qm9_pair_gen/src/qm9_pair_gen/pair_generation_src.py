#!/usr/bin/env python3
"""
Script to generate parquet files for molecule pairs based on Tanimoto distance filtering.

This script:
1. Loads molecular dataset and computes fingerprints
2. For each molecule, computes Tanimoto distance to all other molecules
3. Filters to keep only the closest X percentile of molecules
4. Computes shape Tanimoto distance for the filtered pairs
5. Generates a parquet file with molecule pairs, distances, and shape information
"""

import os
import pickle
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import time
import signal
import random
from datetime import datetime, timedelta
import sys
from omegaconf import DictConfig, OmegaConf

# QM9 and RDKit imports
from atomic_datasets import QM9
from atomic_datasets.datasets.qm9 import get_Anderson_splits
from atomic_datasets.utils.rdkit import is_molecule_sane
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import selfies as sf
import random

# Import our utility functions
from utils_mol import get_shape_tanimoto

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Only show the message, no timestamp/level/logger name
    stream=sys.stdout,  # Explicitly set to stdout
    force=True          # Overwrites any previous logging config
)
logger = logging.getLogger()


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


def timeout_after(seconds):
    """Context manager for timeout functionality."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    
    class TimeoutContext:
        def __enter__(self):
            # Set the signal handler
            self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore the old handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old_handler)
            if exc_type == TimeoutError:
                return True  # Suppress the exception
            return False
    
    return TimeoutContext()


class CustomArrowStreamWriter:
    """Custom Arrow IPC stream writer for molecule pairs with our specific schema.
    Uses record batches for true streaming writes and reads."""
    
    def __init__(self, path, schema):
        self.path = path
        self.schema = schema
        self.row_count = 0
        
        # Ensure the directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create Arrow IPC stream writer for record batches
        self.writer = pa.ipc.new_stream(path, schema)

    def write_chunk(self, chunk):
        """
        chunk: list of (molecule_a_idx, molecule_b_idx, split, morgan_tanimoto_distance, 
                      shape_tanimoto_distance, matchA, matchB, [rmsd, matching_time]) tuples
        """
        batch_data = {
            "molecule_a_idx": [r[0] for r in chunk],
            "molecule_b_idx": [r[1] for r in chunk],
            "split": [r[2] for r in chunk],
            "morgan_tanimoto_distance": [r[3] for r in chunk],
            "shape_tanimoto_distance": [r[4] for r in chunk],
            "matchA": [r[5] for r in chunk],
            "matchB": [r[6] for r in chunk],
        }
        
        # Add extra metrics if they exist in the chunk data
        if len(chunk[0]) > 7:  # Check if extra metrics are present
            batch_data["rmsd"] = [r[7] for r in chunk]
            batch_data["matching_time"] = [r[8] for r in chunk]
        
        # Create record batch (not table) and write to stream
        batch = pa.record_batch(batch_data, schema=self.schema)
        self.writer.write_batch(batch)  # Use write_batch, not write
        self.row_count += batch.num_rows

    def close(self):
        try:
            self.writer.close()
            logger.info(f"✅ Finalized {self.row_count} rows written to {self.path}")
        except Exception as e:
            logger.error(f"Error closing stream writer: {e}")
            # Try to clean up the corrupted file
            if Path(self.path).exists():
                try:
                    Path(self.path).unlink()
                    logger.info(f"Removed corrupted file: {self.path}")
                except Exception as cleanup_error:
                    logger.error(f"Could not remove corrupted file: {cleanup_error}")
            raise


class PairGenerator:
    """
    Generates parquet files for molecule pairs based on Tanimoto distance filtering.
    Supports multiple datasets through configuration.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize the pair generator with configuration.
        
        Args:
            config: Hydra configuration object containing all parameters
        """
        self.config = config
        
        # Log the configuration
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(config))
        
        # Save configuration to output directory
        
        # Extract configuration parameters
        self.cfg = config
        self.data_dir = Path(config.data_dir)
        self.output_dir = Path(config.output_dir)
        self.dataset_name = config.dataset_name
        self.percentile_threshold = config.percentile_threshold
        self.max_molecules = config.get('max_molecules', None)
        self.max_pairs_per_mol = config.get('max_pairs_per_mol', None)
        self.test_mode = config.get('test_mode', False)
        self.test_sizes = config.get('test_sizes', (500, 100, 100))
        self.overwrite_molecules_parquet = config.get('overwrite_molecules_parquet', False)
        self.overwrite_pairs_parquet = config.get('overwrite_pairs_parquet', False)
        self.overwrite_ds = config.get('overwrite_ds', False)
        self.train_only = config.get('train_only', False)
        self.recompute_dataset = config.get('recompute_dataset', False)
        self.use_second_half_of_train = config.get('use_second_half_of_train', False)
        self.log_extra_metrics = config.get('log_extra_metrics', False)  # New config option for logging RMSD and matching time
        self.save_as_parquet = config.get('save_as_parquet', True)  # Whether to convert Arrow IPC to Parquet at the end            

                # Extract splits configuration with validation
        self.splits = config.get('splits', ["train", "val", "test"])
        self._validate_splits()
        
        # Validate pair generation parameters
        self._validate_pair_generation_params()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config_file = os.path.join(self.output_dir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(OmegaConf.to_yaml(config))
        logger.info(f"Configuration record saved to: {config_file}")
        
        
        
        # File paths
        self.molecules_file = self.output_dir / "molecules.parquet"
        self.pairs_file = self.output_dir / "pairs.arrow"  # Use Arrow IPC format for streaming reads
        (Path(self.data_dir) / "cache").mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.split_data = {}
        
        # Statistics tracking
        self.selfies_failures = 0
        self.insane_molecules = 0
        self.valency_issues = 0
        self.total_molecules_processed = 0
        
    def load_dataset(self):
        """Load dataset based on configuration."""
        if self.dataset_name.lower() == "qm9":
            self.load_qm9_dataset()
        elif self.dataset_name.lower() in ["bbbp", "bace", "hiv"]:
            self.load_molecule_net_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Currently only 'qm9' is supported.")
        
    def load_molecule_net_dataset(self):
        """Load MoleculeNet dataset for each split and compute fingerprints, with caching."""
        self.split_data = {}
        ds_cache_dir = Path(self.data_dir) / "cache" / "moleculenet_ds_caches"
        ds_cache_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, split in enumerate(self.dataset_splits):
            logger.info(f"Loading MoleculeNet split '{split}' from {self.data_dir}")
            sdf_path = f'/data/bucket/guided_generation_SM/datasets/moleculenet/{self.dataset_name}_b_{split}_best1.sdf'
            supplier = Chem.SDMolSupplier(sdf_path)
            # Convert to list (filter out invalid molecules)
            molecules = [mol for mol in supplier if mol is not None]
            
            cache_path = ds_cache_dir / f"{self.dataset_name}_preprocessed_{split}.pkl"
            
            if cache_path.exists() and not self.overwrite_ds:
                logger.info(f"Loading split '{split}' data from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                split_dict = cache_data['split_data']
                
                # Take random subset if in test mode
                if self.test_mode:
                    n = self.test_sizes[idx]
                    indices = list(range(len(split_dict['smiles'])))
                    random_indices = random.sample(indices, n)
                    for key in split_dict:
                        if isinstance(split_dict[key], list):
                            split_dict[key] = [split_dict[key][i] for i in random_indices]
                
                # Compute invalid_indices and selfies_failures from loaded data
                invalid_indices = set(i for i, selfies in enumerate(split_dict['selfies']) if selfies is None)
                selfies_failures = sum(1 for selfies in split_dict['selfies'] if selfies is None)
            else:
                if self.test_mode:
                    n = self.test_sizes[idx]
                    molecules = molecules[:n]
                elif self.max_molecules is not None:
                    molecules = molecules[:self.max_molecules]
                    print('maxing out molecules on split', split)
                
                split_dict = {
                    'smiles': [],
                    'selfies': [],
                    'fingerprints': [],
                    'molecules': [],
                    'dataset': molecules
                }

                # Initialize property lists
                property_names = ['SA', 'SC']
                for prop_name in property_names:
                    split_dict[prop_name] = []

                invalid_indices = set()
                selfies_failures = 0
                for i, mol in enumerate(molecules):
                    smiles = Chem.MolToSmiles(mol)
                    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
                    try:
                        selfies = sf.encoder(smiles)
                    except Exception as e:
                        selfies = None
                        selfies_failures += 1
                        invalid_indices.add(i)

                    split_dict['smiles'].append(smiles)
                    split_dict['selfies'].append(selfies)
                    split_dict['fingerprints'].append(fingerprint)
                    split_dict['molecules'].append(mol)
                    
                    # Get SA and SC properties
                    split_dict['SA'].append(float(mol.GetProp('SA')))
                    split_dict['SC'].append(float(mol.GetProp('SC')))

                # Save to cache
                cache_data = {'split_data': split_dict}
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Saved split '{split}' data to {cache_path}")

            # Add computed fields to split_dict
            split_dict['invalid_indices'] = invalid_indices
            split_dict['selfies_failures'] = selfies_failures
            
            self.split_data[split] = split_dict
            logger.info(f"Split '{split}': {len(split_dict['smiles'])} molecules, {selfies_failures} SELFIES failures, {len(invalid_indices)} invalid.")
        
        if self.use_second_half_of_train:
            np.random.seed(42)
            len_train_dset = len(self.split_data['train']['dataset'])
            fixed_perm = np.random.permutation(len_train_dset)
            self.second_half_train_indices = fixed_perm[len_train_dset//2:]
        
    def load_qm9_dataset(self):
        """Load QM9 dataset for each split and compute fingerprints, with caching."""
        self.split_data = {}
        ds_cache_dir = Path(self.data_dir) / "cache" / "qm9_ds_caches"
        ds_cache_dir.mkdir(parents=True, exist_ok=True)
        for idx, split in enumerate(self.dataset_splits):
            logger.info(f"Loading QM9 split '{split}' from {self.data_dir}")
            cache_path = ds_cache_dir / f"qm9_preprocessed_{split}.pkl"
            cache_data = None
            if cache_path.exists() and not self.overwrite_ds:
                logger.info(f"Loading split '{split}' data from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                if 'split_data' in cache_data:
                    split_dict = cache_data['split_data']
                else: # for backwards compatibility
                    split_dict = {
                        'smiles': cache_data['smiles'],
                        'selfies': cache_data['selfies'], 
                        'fingerprints': cache_data['fingerprints'],
                        'molecules': cache_data['molecules'],
                        'dataset': cache_data['dataset']
                    }
                # Compute invalid_indices and selfies_failures from loaded data
                invalid_indices = set(i for i, selfies in enumerate(split_dict['selfies']) if selfies is None)
                selfies_failures = sum(1 for selfies in split_dict['selfies'] if selfies is None)
            else:
                dataset = QM9(root_dir=str(self.data_dir), split=split, check_with_rdkit=False, use_Anderson_splits=True)
                if self.test_mode:
                    n = self.test_sizes[idx]
                    dataset = dataset[:n]
                elif self.max_molecules is not None:
                    dataset = dataset[:self.max_molecules]
                
                split_dict = {
                    'smiles': [],
                    'selfies': [],
                    'fingerprints': [],
                    'molecules': [],
                    'dataset': dataset
                }
                
                property_names = ['gap', 'homo', 'lumo', 'u0', 'u', 'h', 'g', 'cv', 'mu', 'alpha', 'r2', 'zpve']
                for prop_name in property_names:
                    split_dict[prop_name] = []
                
                invalid_indices = set()
                selfies_failures = 0
                for i, graph in enumerate(dataset):
                    mol = graph['properties']['rdkit_mol']
                    smiles = Chem.MolToSmiles(mol)
                    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
                    try:
                        selfies = sf.encoder(smiles)
                    except Exception as e:
                        selfies = None
                        invalid_indices.add(i)
                        selfies_failures += 1
                    for prop_name in property_names:
                        value = graph['properties'].get(prop_name, 0.0)
                        if np.isnan(value):
                            value = 0.0
                        split_dict[prop_name].append(float(value))
                    split_dict['smiles'].append(smiles)
                    split_dict['selfies'].append(selfies)
                    split_dict['fingerprints'].append(fingerprint)
                    split_dict['molecules'].append(mol)
                
                cache_data = {'split_data': split_dict}
                with open(cache_path, "wb") as f:
                    pickle.dump(cache_data, f)
                logger.info(f"Saved split '{split}' data to {cache_path}")
            
            # Add computed fields to split_dict
            split_dict['invalid_indices'] = invalid_indices
            split_dict['selfies_failures'] = selfies_failures
            
            self.split_data[split] = split_dict
            logger.info(f"Split '{split}': {len(split_dict['smiles'])} molecules, {selfies_failures} SELFIES failures, {len(invalid_indices)} invalid.")
        
        if self.use_second_half_of_train:
            np.random.seed(42)
            len_train_dset = len(self.split_data['train']['dataset'])
            fixed_perm = np.random.permutation(len_train_dset)
            self.second_half_train_indices = fixed_perm[len_train_dset//2:]
        
    def _validate_splits(self):
        """Validate the splits configuration."""
        # Map user-friendly split names to dataset split names
        split_mapping = {
            "train": "train",
            "val": "val",  
            "test": "test"
        }
        
        # Handle both regular lists and Hydra's ListConfig
        from omegaconf import ListConfig
        
        if not (isinstance(self.splits, list) or isinstance(self.splits, ListConfig)):
            raise ValueError(f"splits must be a list, got {type(self.splits)}")
        
        # Convert to regular list if it's a ListConfig
        splits_list = list(self.splits) if isinstance(self.splits, ListConfig) else self.splits
        
        for split in splits_list:
            if split not in split_mapping:
                raise ValueError(f"Invalid split '{split}'. Valid splits are: {list(split_mapping.keys())}")
        
        if not splits_list:
            raise ValueError("splits list cannot be empty")
        
        print("splits_list", splits_list)
        
        # Update self.splits to be a regular list and create dataset splits mapping
        self.splits = splits_list
        self.dataset_splits = [split_mapping[split] for split in splits_list]
        
        logger.info(f"✅ Using splits: {self.splits} (maps to dataset splits: {self.dataset_splits})")
        
    def _format_time_estimate(self, seconds: float) -> str:
        """Format time estimate in human readable format (days, hours, minutes)."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
        else:
            days = seconds / 86400
            return f"{days:.1f} days"
        
    def generate_molecules_parquet(self):
        """Generate the molecules parquet file for all specified splits."""
        # Check if file exists and recomputation not forced
        if self.molecules_file.exists() and not self.overwrite_molecules_parquet:
            logger.info(f"Molecules file {self.molecules_file} already exists. Use overwrite_molecules_parquet=True to regenerate.")
            return
            
        logger.info(f"Generating molecules parquet file for splits: {self.splits}")
        
        # Determine property names based on dataset
        if self.dataset_name.lower() == "qm9":
            property_names = ['gap', 'homo', 'lumo', 'u0', 'u', 'h', 'g', 'cv', 'mu', 'alpha', 'r2', 'zpve']
        else:  # MoleculeNet datasets
            property_names = ['SA', 'SC']
        
        # Create molecules data for all specified splits
        all_data = []
        self.split_valid_indices = {}  # Store valid indices for each split
        self.split_valid_indices_to_new_idx = {}  # Store mapping for each split
        
        global_item_id = 0  # Global counter across all splits
        
        for user_split, dataset_split in zip(self.splits, self.dataset_splits):
            if dataset_split not in self.split_data:
                raise ValueError(f"Split '{dataset_split}' not found in loaded data. Available splits: {list(self.split_data.keys())}")
            
            split_data = self.split_data[dataset_split]
            
            # Get valid molecules (exclude invalid ones)
            valid_indices = [i for i in range(len(split_data['smiles'])) if i not in split_data['invalid_indices']]
            logger.info(f"Split '{dataset_split}': {len(valid_indices)} valid molecules, {len(split_data['invalid_indices'])} invalid.")
            
            # Store valid indices for this split
            self.split_valid_indices[dataset_split] = valid_indices
            
            # Create mapping for this split
            split_mapping = {}
            
            # Add molecules for this split
            for original_idx in valid_indices:
                row = {
                    'item_id': global_item_id,  # Global sequential indexing
                    'smiles': split_data['smiles'][original_idx],
                    'selfies': split_data['selfies'][original_idx],
                    'split': user_split  # Use user-friendly split name
                }
                for prop_name in property_names:
                    if prop_name in split_data:
                        row[prop_name] = split_data[prop_name][original_idx]
                    else:
                        row[prop_name] = 0.0  # Default value if property not found
                all_data.append(row)
                
                # Store mapping from original index to global item_id
                split_mapping[original_idx] = global_item_id
                global_item_id += 1
            
            self.split_valid_indices_to_new_idx[dataset_split] = split_mapping
        
        molecules_df = pl.DataFrame(all_data)
        molecules_df.write_parquet(self.molecules_file)
        logger.info(f"Saved molecules to {self.molecules_file} (shape: {molecules_df.shape})")
        
    def _validate_pair_generation_params(self):
        """Validate that only one of percentile_threshold or max_pairs_per_mol is specified."""
        percentile_specified = self.percentile_threshold is not None
        max_pairs_specified = self.max_pairs_per_mol is not None
        
        if percentile_specified and max_pairs_specified:
            raise ValueError(
                f"Both percentile_threshold ({self.percentile_threshold}) and max_pairs_per_mol ({self.max_pairs_per_mol}) "
                f"are specified. Please specify only one of them."
            )
        
        if not percentile_specified and not max_pairs_specified:
            raise ValueError(
                "Neither percentile_threshold nor max_pairs_per_mol is specified. "
                "Please specify one of them."
            )
        
        if percentile_specified:
            if self.percentile_threshold <= 0 or self.percentile_threshold > 100:
                raise ValueError(f"percentile_threshold must be between 0 and 100, got {self.percentile_threshold}")
            logger.info(f"✅ Using percentile_threshold: {self.percentile_threshold}%")
        
        if max_pairs_specified:
            if self.max_pairs_per_mol <= 0:
                raise ValueError(f"max_pairs_per_mol must be positive, got {self.max_pairs_per_mol}")
            logger.info(f"✅ Using max_pairs_per_mol: {self.max_pairs_per_mol}")

    def compute_tanimoto_distances_and_pairs(self):
        """Compute Tanimoto distances and generate pairs for all specified splits."""
        
        # Determine the method and log it
        if self.percentile_threshold is not None:
            logger.info(f"Computing Tanimoto distances and filtering to {self.percentile_threshold}% closest pairs for splits: {self.splits}")
        else:
            logger.info(f"Computing Tanimoto distances and keeping {self.max_pairs_per_mol} closest pairs per molecule for splits: {self.splits}")
        
        # Define schema for the pairs parquet file (including split column)
        schema_fields = [
            ("molecule_a_idx", pa.int32()),
            ("molecule_b_idx", pa.int32()),
            ("split", pa.string()),  # Add split column
            ("morgan_tanimoto_distance", pa.float32()),
            ("shape_tanimoto_distance", pa.float32()),
            ("matchA", pa.list_(pa.int32())),
            ("matchB", pa.list_(pa.int32())),
        ]
        
        # Add extra metrics fields if enabled
        if self.log_extra_metrics:
            schema_fields.extend([
                ("rmsd", pa.float32()),
                ("matching_time", pa.float32()),
            ])
        
        schema = pa.schema(schema_fields)
        
        # Initialize stream writer
        stream_writer = None
        try:
            stream_writer = CustomArrowStreamWriter(str(self.pairs_file), schema)
            chunk_results = []
            write_every = self.cfg.get('write_every', 10000)
            
            # Time tracking variables
            start_time = time.time()
            last_status_time = start_time
            status_interval = 60
            total_pairs_processed = 0
            
            # Process each split
            for user_split, dataset_split in zip(self.splits, self.dataset_splits):
                logger.info(f"Processing split: {user_split} (dataset: {dataset_split})")
                
                # Get data for this split
                split_data = self.split_data[dataset_split]
                smiles_list = split_data['smiles']
                fingerprints = split_data['fingerprints']
                molecules = split_data['molecules']
                
                # Use the valid indices that were computed in generate_molecules_parquet
                if not hasattr(self, 'split_valid_indices'):
                    raise ValueError("Valid indices not found. Run generate_molecules_parquet() first.")
                
                valid_indices = self.split_valid_indices[dataset_split]
                
                # Apply second half of train filtering if enabled
                if self.use_second_half_of_train and dataset_split == "train":
                    if hasattr(self, 'second_half_train_indices'):
                        # Filter valid_indices to only include molecules in the second half
                        valid_indices = [idx for idx in valid_indices if idx in self.second_half_train_indices]
                        logger.info(f"Using second half of train split: {len(valid_indices)} molecules")
                    else:
                        logger.warning("second_half_train_indices not found, using all valid molecules")
                
                total_molecules = len(valid_indices)
                
                # Calculate pairs per molecule based on the specified method
                if self.percentile_threshold is not None:
                    # Use percentile method
                    percentile_decimal = self.percentile_threshold / 100.0
                    pairs_per_molecule = max(1, int((total_molecules - 1) * percentile_decimal))
                else:
                    # Use max_pairs_per_mol method
                    pairs_per_molecule = min(self.max_pairs_per_mol, total_molecules - 1)
                
                total_expected_pairs = total_molecules * pairs_per_molecule
                
                logger.info(f"Split '{user_split}': {total_expected_pairs} expected pairs, {pairs_per_molecule} pairs per molecule")
                logger.info(f"Processing {len(valid_indices)} valid molecules in split '{dataset_split}'")
                
                # Process each molecule in this split
                for idx, original_idx in enumerate(tqdm(valid_indices, desc=f"Processing molecule pairs in split '{user_split}'")):
                    mol_a = molecules[original_idx]
                    fp_a = fingerprints[original_idx]
                    
                    # Create list of other valid molecules for comparison (same split only)
                    other_fps = []
                    other_indices = []
                    other_mols = []
                    
                    for other_original_idx in valid_indices:
                        if other_original_idx != original_idx:  # Skip self-comparison
                            other_fps.append(fingerprints[other_original_idx])
                            other_indices.append(other_original_idx)
                            other_mols.append(molecules[other_original_idx])
                    
                    # Compute Tanimoto similarities in bulk
                    tanimoto_similarities = DataStructs.BulkTanimotoSimilarity(fp_a, other_fps)
                    
                    # Convert to distances and create distance list
                    distances = []
                    for j_idx, similarity in enumerate(tanimoto_similarities):
                        tanimoto_distance = 1.0 - similarity
                        distances.append((other_indices[j_idx], tanimoto_distance, other_mols[j_idx]))
                    
                    # Sort by distance (closest first)
                    distances.sort(key=lambda x: x[1])
                    
                    # Keep only the closest pairs based on the specified method
                    closest_pairs = distances[:pairs_per_molecule]
                    
                    # Compute shape Tanimoto distance for the closest pairs
                    for other_original_idx, morgan_distance, mol_b in closest_pairs:
                        try:
                            timeout = self.config.get('shape_tanimoto_timeout', 10.0)
                            with timeout_after(timeout):
                                shape_distance, matchA, matchB, rmsd, matching_time = get_shape_tanimoto(mol_a, mol_b, return_extra=True)
                        except TimeoutError:
                            logger.warning(f"Shape Tanimoto computation timed out after {timeout}s")
                            shape_distance, matchA, matchB, rmsd, matching_time = -1.0, [], [], -1.0, -1.0
                        
                        # Use the global indices from molecules parquet
                        molecule_a_idx = self.split_valid_indices_to_new_idx[dataset_split][original_idx]
                        molecule_b_idx = self.split_valid_indices_to_new_idx[dataset_split][other_original_idx]
                        
                        # Create pair entry with split information
                        pair_entry = (
                            molecule_a_idx,
                            molecule_b_idx,
                            user_split,  # Add split column
                            float(morgan_distance),
                            float(shape_distance),
                            matchA,
                            matchB
                        )
                        
                        # Add extra metrics if enabled
                        if self.log_extra_metrics:
                            pair_entry += (float(rmsd), float(matching_time))
                        
                        chunk_results.append(pair_entry)
                    
                    # Update total pairs processed
                    total_pairs_processed += len(closest_pairs)
                    
                    # Write chunks periodically
                    if len(chunk_results) >= write_every:
                        stream_writer.write_chunk(chunk_results)
                        chunk_results = []
                    
                    # Print status periodically
                    current_time = time.time()
                    if current_time - last_status_time >= status_interval:
                        elapsed_time = current_time - start_time
                        
                        if total_pairs_processed > 0:
                            pairs_per_second = total_pairs_processed / elapsed_time
                            
                            status_msg = (
                                f"STATUS UPDATE - "
                                f"Total pairs: {total_pairs_processed:,} | "
                                f"Rate: {pairs_per_second:.1f} pairs/sec | "
                                f"Elapsed: {self._format_time_estimate(elapsed_time)}"
                            )
                        else:
                            status_msg = (
                                f"STATUS UPDATE - "
                                f"Starting pair generation... | "
                                f"Elapsed: {self._format_time_estimate(elapsed_time)}"
                            )
                        
                        logger.info(status_msg)
                        last_status_time = current_time
            
            # Write any remaining chunks
            if chunk_results:
                stream_writer.write_chunk(chunk_results)
            
            # Close the stream writer
            stream_writer.close()
            
            logger.info(f"✅ Pair generation completed! Total pairs: {total_pairs_processed:,}")
            
        except Exception as e:
            logger.error(f"❌ Error during pair generation: {e}")
            if stream_writer:
                stream_writer.close()
            raise e

    def generate_pairs_parquet(self):
        """Generate the pairs Arrow IPC file using stream writing."""
        logger.info("Generating pairs Arrow IPC file...")
        
        # Compute distances and pairs using stream writing
        self.compute_tanimoto_distances_and_pairs()
        
        logger.info(f"Saved pairs to {self.pairs_file}")
        
        # Read the file to get statistics
        try:
            # Try reading as Arrow IPC format            
            reader = pa.ipc.RecordBatchStreamReader(self.pairs_file)
            table = reader.read_all()
            pairs_df = pl.from_arrow(table)
        except Exception as e:
            logger.warning(f"Could not read Arrow IPC file: {e}")
            logger.info("File may still be in progress or in a different format")
            return
            
        logger.info(f"Pairs dataframe shape: {pairs_df.shape}")
        logger.info(f"Columns: {pairs_df.columns}")
        
        # Print some statistics
        if len(pairs_df) > 0:
            morgan_distances = pairs_df['morgan_tanimoto_distance'].to_list()
            shape_distances = [d for d in pairs_df['shape_tanimoto_distance'].to_list() if not np.isnan(d)]
            
            logger.info(f"Morgan Tanimoto distance stats:")
            logger.info(f"  Mean: {np.mean(morgan_distances):.4f}")
            logger.info(f"  Std: {np.std(morgan_distances):.4f}")
            logger.info(f"  Min: {np.min(morgan_distances):.4f}")
            logger.info(f"  Max: {np.max(morgan_distances):.4f}")
            
            if shape_distances:
                logger.info(f"Shape Tanimoto distance stats:")
                logger.info(f"  Mean: {np.mean(shape_distances):.4f}")
                logger.info(f"  Std: {np.std(shape_distances):.4f}")
                logger.info(f"  Min: {np.min(shape_distances):.4f}")
                logger.info(f"  Max: {np.max(shape_distances):.4f}")
        
    def read_pairs_file_streaming(self):
        """Read the pairs Arrow IPC file while it's being written.
        Returns a Polars DataFrame if the file can be read, None otherwise."""
        try:
            # Try reading as Arrow IPC format
            pairs_df = pl.read_ipc(self.pairs_file)
            return pairs_df
        except Exception as e:
            # Check if file exists but is empty or incomplete
            if self.pairs_file.exists():
                file_size = self.pairs_file.stat().st_size
                if file_size == 0:
                    logger.debug("Arrow IPC file exists but is empty")
                else:
                    logger.debug(f"Arrow IPC file exists (size: {file_size}) but cannot be read: {e}")
            else:
                logger.debug(f"Arrow IPC file does not exist yet: {e}")
            return None
    
    def get_pairs_file_stats(self):
        """Get statistics about the pairs file if it can be read."""
        pairs_df = self.read_pairs_file_streaming()
        if pairs_df is None:
            return None
            
        stats = {
            'total_rows': len(pairs_df),
            'columns': pairs_df.columns,
        }
        
        if len(pairs_df) > 0:
            morgan_distances = pairs_df['morgan_tanimoto_distance'].to_list()
            shape_distances = [d for d in pairs_df['shape_tanimoto_distance'].to_list() if not np.isnan(d)]
            
            stats['morgan_distance'] = {
                'mean': float(np.mean(morgan_distances)),
                'std': float(np.std(morgan_distances)),
                'min': float(np.min(morgan_distances)),
                'max': float(np.max(morgan_distances)),
            }
            
            if shape_distances:
                stats['shape_distance'] = {
                    'mean': float(np.mean(shape_distances)),
                    'std': float(np.std(shape_distances)),
                    'min': float(np.min(shape_distances)),
                    'max': float(np.max(shape_distances)),
                }
        
        return stats
        
    def convert_to_parquet(self):
        """Convert the Arrow IPC file to Parquet format for compatibility."""
        if not self.pairs_file.exists():
            logger.warning(f"Arrow IPC file {self.pairs_file} does not exist")
            return
            
        parquet_file = self.output_dir / "pairs.parquet"
        try:
            # Try reading with PyArrow first, then convert to Polars
            with pa.ipc.open_stream(self.pairs_file) as reader:
                batches = []
                try:
                    while True:
                        batch = reader.read_next_batch()
                        if batch is None:
                            break
                        batches.append(batch)
                except StopIteration:
                    pass
                
                if batches:
                    # Convert record batches to table and concatenate
                    # Use pa.Table.from_batches() instead of individual to_table() calls
                    try:
                        #breakpoint()
                        table = pa.Table.from_batches(batches)
                        # Convert to Polars DataFrame
                        pairs_df = pl.from_arrow(table)
                        pairs_df.write_parquet(parquet_file)
                        logger.info(f"✅ Converted Arrow IPC to Parquet: {parquet_file}")
                    except Exception as e:
                        logger.error(f"❌ Error converting Arrow IPC to Parquet: {e}")
                        raise e
                else:
                    logger.warning("No data found in Arrow IPC file")
                    
        except Exception as e:
            logger.error(f"Failed to convert Arrow IPC to Parquet: {e}")
            # Try alternative method using Polars directly
            try:
                pairs_df = pl.read_ipc(self.pairs_file)
                pairs_df.write_parquet(parquet_file)
                logger.info(f"✅ Converted Arrow IPC to Parquet (alternative method): {parquet_file}")
            except Exception as e2:
                logger.error(f"Alternative conversion method also failed: {e2}")
    
    def generate_parquet_files(self):
        """Generate all parquet files."""
        logger.info("Starting parquet file generation...")
        
        # Check if files already exist
        if not self.overwrite_pairs_parquet and (self.molecules_file.exists() or self.pairs_file.exists()):
            logger.warning("Files already exist. Use overwrite_parquet=True to regenerate.")
            return  
        
        # Load dataset
        self.load_dataset()
        
        # Generate parquet files
        self.generate_molecules_parquet()
        self.generate_pairs_parquet()  # No longer needs splits parameter
        
        # Convert to Parquet format if requested
        if self.save_as_parquet:
            self.convert_to_parquet()
        
        logger.info("Parquet file generation completed!")


def main():
    """Legacy main function for backward compatibility."""
    print("This script has been updated to use Hydra configuration.")
    print("Please use the new generate_pairs.py script instead:")
    print("python generate_pairs.py")
    print("\nOr run with custom configuration:")
    print("python generate_pairs.py dataset_name=qm9 percentile_threshold=10 test_mode=true")


if __name__ == "__main__":
    main() 

# to-do's (do not delete this comment)
# fix splits to use anderson splits - now done
# track for how many the selfies computation fails (or the molecule is not sane) and print a statistic on this