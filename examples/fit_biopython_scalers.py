"""
Fit StandardScalers to BioPython features for a given dataset
and save the scaler parameters (mean and scale) to a JSON file.

For protein features, the scaler parameters were obtained by fitting 
a StandardScaler to the protein features of the AMPLIFY dataset
and peptide parameters were obtained from PeptideAtlas.


Example:
python fit_biopython_scalers.py AMPLIFY --num-samples 50000 --num-val-samples 10000 
"""

import json
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from lobster.constants import BIOPYTHON_FEATURES
from lobster.data import UMELightningDataModule
from lobster.transforms.functional import get_biopython_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_sequence_batch(args: tuple[list[str], list[str], int]) -> tuple[list[dict[str, float]], int]:
    """Process a batch of sequences and extract BioPython features.
    
    Parameters
    ----------
    args : tuple
        Tuple containing (sequences, feature_list, worker_id)
        
    Returns
    -------
    tuple[list[dict[str, float]], int]
        List of feature dictionaries and number of failed sequences
    """
    sequences, feature_list, worker_id = args
    results = []
    failed_count = 0
    
    for sequence in sequences:
        try:
            features = get_biopython_features(
                sequence,
                feature_list=feature_list,
                return_as_tensor=False
            )
            
            if features is None:
                failed_count += 1
                continue
                
            results.append(features)
            
        except Exception as e:
            logger.debug(f"Worker {worker_id}: Failed to process sequence: {e}")
            failed_count += 1
            continue
    
    return results, failed_count


def fit_scaler(data: np.ndarray, feature_name: str) -> StandardScaler | None:
    """Fit a StandardScaler to feature data.
    
    Parameters
    ----------
    data : np.ndarray
        Feature values to fit
    feature_name : str
        Name of the feature for logging
        
    Returns
    -------
    StandardScaler | None
        Fitted scaler or None if insufficient valid data
    """
    # Remove NaN and infinite values
    clean_data = data[np.isfinite(data)]
    
    if len(clean_data) < 100:
        logger.warning(f"Insufficient valid data for {feature_name}: {len(clean_data)} samples")
        return None
    
    scaler = StandardScaler()
    scaler.fit(clean_data.reshape(-1, 1))
    
    logger.info(f"Fitted {feature_name}: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
    
    return scaler


def validate_scaler(scaler: StandardScaler, validation_data: np.ndarray) -> bool:
    """Validate scaler quality on validation data.
    
    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler to validate
    validation_data : np.ndarray
        Validation data to test scaler quality
        
    Returns
    -------
    bool
        True if scaler shows good fit, False otherwise
    """
    clean_data = validation_data[np.isfinite(validation_data)]
    
    if len(clean_data) < 10:
        return False
    
    try:
        scaled_data = scaler.transform(clean_data.reshape(-1, 1)).flatten()
        
        # Good fit criteria: scaled mean ~0 and std ~1
        scaled_mean = np.mean(scaled_data)
        scaled_std = np.std(scaled_data)
        
        mean_ok = abs(scaled_mean) < 0.5
        std_ok = abs(scaled_std - 1.0) < 0.5
        outlier_ok = np.percentile(np.abs(scaled_data), 99) < 5.0
        
        return mean_ok and std_ok and outlier_ok
        
    except Exception:
        return False


def collect_sequences_from_dataloader(dataloader, num_samples: int, desc: str) -> list[str]:
    """Efficiently collect sequences using dataloader batching.
    
    Parameters
    ----------
    dataloader
        PyTorch dataloader
    num_samples : int
        Number of sequences to collect
    desc : str
        Description for progress bar
        
    Returns
    -------
    list[str]
        List of collected sequences
    """
    sequences = []
    
    with tqdm(desc=desc, total=num_samples) as pbar:
        for batch in dataloader:
            batch_sequences = batch['sequence']
            
            # Handle both single sequences and batches
            if isinstance(batch_sequences, str):
                sequences.append(batch_sequences)
                pbar.update(1)
            else:
                for seq in batch_sequences:
                    sequences.append(seq)
                    pbar.update(1)
                    if len(sequences) >= num_samples:
                        break
            
            if len(sequences) >= num_samples:
                break
    
    return sequences[:num_samples]


def process_sequences_parallel(sequences: list[str], num_workers: int, batch_size: int, desc: str) -> tuple[dict[str, list[float]], int]:
    """Process sequences using multiprocessing.
    
    Parameters
    ----------
    sequences : list[str]
        List of sequences to process
    num_workers : int
        Number of worker processes
    batch_size : int
        Batch size for processing
    desc : str
        Description for progress bar
        
    Returns
    -------
    tuple[dict[str, list[float]], int]
        Feature data dictionary and number of failed sequences
    """
    feature_data = defaultdict(list)
    total_failed = 0
    
    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batches.append((batch, list(BIOPYTHON_FEATURES), i // batch_size))
    
    logger.info(f"Processing {len(sequences)} sequences in {len(batches)} batches using {num_workers} workers")
    
    # Process in parallel
    with mp.Pool(num_workers) as pool:
        batch_results = list(tqdm(
            pool.imap(process_sequence_batch, batches),
            total=len(batches),
            desc=desc
        ))
    
    # Aggregate results
    for batch_features_list, batch_failed in batch_results:
        total_failed += batch_failed
        
        for features in batch_features_list:
            for feature_name, feature_value in features.items():
                if np.isfinite(feature_value):
                    feature_data[feature_name].append(feature_value)
    
    return dict(feature_data), total_failed


def create_feature_dataframe(feature_data: dict[str, list[float]]) -> pd.DataFrame:
    """Create DataFrame from feature data dictionary.
    
    Parameters
    ----------
    feature_data : dict[str, list[float]]
        Dictionary mapping feature names to lists of values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with features as columns
    """
    df_data = {}
    for feature_name in BIOPYTHON_FEATURES:
        if feature_name in feature_data:
            df_data[feature_name] = feature_data[feature_name]
        else:
            logger.warning(f"No data collected for feature: {feature_name}")
    
    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df_data.items()]))


def _setup_environment(random_seed: int, num_workers: int | None, output_dir: str | None) -> tuple[int, Path | None]:
    """Setup random seeds, workers, and output directory."""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    output_path = None
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)
    
    return num_workers, output_path


def _create_data_modules(dataset_name: str, dataloader_batch_size: int) -> tuple[UMELightningDataModule, UMELightningDataModule]:
    """Create and setup train and validation data modules."""
    train_dm = UMELightningDataModule(
        datasets=[dataset_name],
        root="/data2/ume/.cache2/",
        max_length=1024,
        batch_size=dataloader_batch_size,
    )
    train_dm.setup("fit")
    
    val_dm = UMELightningDataModule(
        datasets=[dataset_name],
        root="/data2/ume/.cache2/",
        max_length=1024,
        batch_size=dataloader_batch_size,
    )
    val_dm.setup("validate")
    
    return train_dm, val_dm


def _fit_and_validate_scalers(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict[str, StandardScaler], dict[str, bool]]:
    """Fit scalers on training data and validate on validation data."""
    scalers = {}
    validation_results = {}
    
    for feature_name in BIOPYTHON_FEATURES:
        if feature_name not in train_df.columns:
            continue
            
        train_data = train_df[feature_name].dropna().values
        scaler = fit_scaler(train_data, feature_name)
        
        if scaler is not None:
            scalers[feature_name] = scaler
            
            # Validate on validation data
            if feature_name in val_df.columns:
                val_data = val_df[feature_name].dropna().values
                is_valid = validate_scaler(scaler, val_data)
                validation_results[feature_name] = is_valid
                
                if not is_valid:
                    logger.warning(f"{feature_name}: Poor fit on validation data")
    
    return scalers, validation_results


def _save_results(scalers: dict[str, StandardScaler], dataset_name: str, output_path: Path | None, results: dict) -> None:
    """Save scaler parameters and results to files."""
    if not output_path:
        return
    
    # Save scaler parameters as JSON
    scaler_params = {
        feature_name: {
            'mean': float(scaler.mean_[0]),
            'scale': float(scaler.scale_[0])
        }
        for feature_name, scaler in scalers.items()
    }
    
    json_file = output_path / f"{dataset_name}_biopython_scaler_params.json"
    with open(json_file, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    logger.info(f"Saved scaler parameters to {json_file}")
    
    # Save full results
    results_file = output_path / f"{dataset_name}_scaler_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")


def _print_results(dataset_name: str, scalers: dict[str, StandardScaler], validation_results: dict[str, bool]) -> None:
    """Print scaler results to console."""
    logger.info(f"\nSCALER RESULTS FOR {dataset_name.upper()}:")
    logger.info(f"Successfully fitted {len(scalers)} scalers")
    
    for feature_name, scaler in scalers.items():
        validation_status = "✓" if validation_results.get(feature_name, True) else "⚠"
        mean = scaler.mean_[0]
        scale = scaler.scale_[0]
        logger.info(f"  {validation_status} {feature_name}: mean={mean:.4f}, std={scale:.4f}")


def analyze_biopython_scalers(
    dataset_name: str,
    num_samples: int = 50_000, 
    num_validation_samples: int = 10_000,
    random_seed: int = 42,
    output_dir: str | None = "biopython_scalers",
    num_workers: int | None = None,
    batch_size: int = 1000,
    dataloader_batch_size: int = 64
) -> dict[str, Any]:
    """Analyze BioPython features and fit StandardScalers using multiprocessing.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to analyze
    num_samples : int, default=50_000
        Number of sequences for training scalers
    num_validation_samples : int, default=10_000
        Number of sequences for validation
    random_seed : int, default=42
        Random seed for reproducibility
    output_dir : str | None, default="biopython_scalers"
        Output directory path
    num_workers : int | None, default=None
        Number of worker processes (defaults to CPU count)
    batch_size : int, default=1000
        Batch size for multiprocessing
    dataloader_batch_size : int, default=64
        Batch size for dataloader
        
    Returns
    -------
    dict[str, Any]
        Analysis results including scaler parameters and metadata
    """
    # Setup environment
    num_workers, output_path = _setup_environment(random_seed, num_workers, output_dir)
    
    logger.info(f"Analyzing {dataset_name} dataset with {num_workers} workers")
    logger.info(f"Training samples: {num_samples}, Validation samples: {num_validation_samples}")
    logger.info(f"Dataloader batch size: {dataloader_batch_size}, Processing batch size: {batch_size}")
    if output_path:
        logger.info(f"Output directory: {output_path}")
    
    # Create data modules
    train_dm, val_dm = _create_data_modules(dataset_name, dataloader_batch_size)
    
    # Collect sequences
    logger.info("Collecting training sequences...")
    train_sequences = collect_sequences_from_dataloader(
        train_dm.train_dataloader(), num_samples, "Collecting training sequences"
    )
    
    logger.info("Collecting validation sequences...")
    val_sequences = collect_sequences_from_dataloader(
        val_dm.val_dataloader(), num_validation_samples, "Collecting validation sequences"
    )
    
    logger.info(f"Collected {len(train_sequences)} training + {len(val_sequences)} validation sequences")
    
    # Process sequences
    logger.info("Processing training sequences...")
    train_features, train_failed = process_sequences_parallel(
        train_sequences, num_workers, batch_size, "Training sequences"
    )
    
    logger.info("Processing validation sequences...")
    val_features, val_failed = process_sequences_parallel(
        val_sequences, num_workers, batch_size, "Validation sequences"
    )
    
    total_failed = train_failed + val_failed
    logger.info(f"Processing complete. Failed sequences: {total_failed}")
    
    # Create DataFrames
    train_df = create_feature_dataframe(train_features)
    val_df = create_feature_dataframe(val_features)
    
    # Save DataFrames
    if output_path:
        train_df.to_parquet(output_path / f"{dataset_name}_train_features.parquet")
        val_df.to_parquet(output_path / f"{dataset_name}_val_features.parquet")
        logger.info(f"Saved feature DataFrames to {output_path}")
    
    # Print statistics
    logger.info(f"\n{dataset_name.upper()} TRAINING FEATURE STATISTICS (n={len(train_sequences)})")
    print(train_df.describe().round(4))
    
    # Fit and validate scalers
    logger.info("\nFitting scalers...")
    scalers, validation_results = _fit_and_validate_scalers(train_df, val_df)
    
    # Extract scaler parameters
    scaler_params = {
        feature_name: {
            'mean': float(scaler.mean_[0]),
            'scale': float(scaler.scale_[0])
        }
        for feature_name, scaler in scalers.items()
    }
    
    # Print results
    _print_results(dataset_name, scalers, validation_results)
    
    # Prepare results dictionary
    results = {
        'dataset_name': dataset_name,
        'scaler_params': scaler_params,
        'validation_results': validation_results,
        'statistics': train_df.describe().to_dict(),
        'metadata': {
            'num_training_samples': len(train_sequences),
            'num_validation_samples': len(val_sequences),
            'failed_sequences': total_failed,
            'random_seed': random_seed,
            'num_workers': num_workers,
            'batch_size': batch_size,
            'dataloader_batch_size': dataloader_batch_size,
            'output_path': str(output_path) if output_path else None
        }
    }
    
    # Save results
    _save_results(scalers, dataset_name, output_path, results)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze BioPython features and fit StandardScalers")
    parser.add_argument("dataset", help="Dataset name (e.g., AMPLIFY, PeptideAtlas)")
    parser.add_argument("--num-samples", type=int, default=50_000, help="Training samples")
    parser.add_argument("--num-val-samples", type=int, default=10_000, help="Validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--num-workers", type=int, help="Number of workers (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Processing batch size")
    parser.add_argument("--dataloader-batch-size", type=int, default=64, help="Dataloader batch size")

    args = parser.parse_args()
    
    analyze_biopython_scalers(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        num_validation_samples=args.num_val_samples,
        random_seed=args.seed,
        output_dir=args.output,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        dataloader_batch_size=args.dataloader_batch_size
    )