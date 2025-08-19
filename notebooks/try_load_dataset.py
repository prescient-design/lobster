#!/usr/bin/env python3
"""
Script to load the datamodule based on the given hydra config and input arguments.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import hydra.core.global_hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lobster.data._molecule_improvement_datamodule import MoleculeImprovementLightningDataModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load datamodule with hydra config")
    parser.add_argument(
        "--config-path", 
        type=str, 
        default="src/lobster/hydra_config/train_molecule_improvement.yaml",
        help="Path to the hydra config file"
    )
    parser.add_argument(
        "--overrides", 
        nargs="*", 
        default=[],
        help="Hydra overrides (e.g., data.batch_size=8 data.root=/path/to/data)"
    )
    parser.add_argument(
        "--stage", 
        type=str, 
        default="fit",
        choices=["fit", "test", "predict"],
        help="Setup stage for the datamodule"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Don't actually load data, just print config"
    )
    return parser.parse_args()

def find_latest_ckpt_file(directory, use_val_loss=False):
    """
    Find either the most recently modified checkpoint file (.ckpt) or the one with lowest val_loss
    in a directory and its subdirectories.

    Args:
        directory (str): The root directory to search for checkpoint files.
        use_val_loss (bool): If True, find checkpoint with lowest val_loss. If False, use most recent.

    Returns:
        str or None: Path to the checkpoint file, or None if no checkpoint files are found.
    """
    if not use_val_loss:
        latest_ckpt_path = None
        latest_mtime = -1

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.ckpt'):
                    full_path = os.path.join(root, file)
                    mtime = os.path.getmtime(full_path)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_ckpt_path = full_path

        return latest_ckpt_path

    else:
        best_ckpt_path = None
        lowest_val_loss = float('inf')

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.ckpt'):
                    try:
                        # Extract val_loss from filename like epoch=6-step=191112-val_loss=0.4235.ckpt
                        val_loss = float(file.split('val_loss=')[-1].split('.ckpt')[0])
                        full_path = os.path.join(root, file)
                        if val_loss < lowest_val_loss:
                            lowest_val_loss = val_loss
                            best_ckpt_path = full_path
                    except (IndexError, ValueError):
                        # Skip files that don't match the expected format
                        continue

        return best_ckpt_path

def deep_compare(obj1, obj2, visited=None, path="root", fail_fast=True):
    """
    Recursively compares two objects by their attributes.
    If fail_fast is False, prints all mismatches before returning.
    """
    if visited is None:
        visited = set()

    # Track differences
    differences_found = False

    # Avoid cycles
    if (id(obj1), id(obj2)) in visited:
        return True
    visited.add((id(obj1), id(obj2)))

    # Base case: same value
    if obj1 == obj2:
        return True

    # Type mismatch
    if type(obj1) != type(obj2):
        print(f"Type mismatch at {path}: {type(obj1)} != {type(obj2)}")
        return not fail_fast or False

    # Dicts
    if isinstance(obj1, dict):
        keys1, keys2 = obj1.keys(), obj2.keys()
        if keys1 != keys2:
            print(f"Key mismatch at {path}: {keys1} != {keys2}")
            if fail_fast:
                return False
            differences_found = True
        for k in obj1:
            if k in obj2:
                result = deep_compare(obj1[k], obj2[k], visited, f"{path}[{repr(k)}]", fail_fast)
                if not result:
                    if fail_fast:
                        return False
                    differences_found = True
        return not differences_found

    # Sequences
    if isinstance(obj1, (list, tuple, set)):
        if len(obj1) != len(obj2):
            print(f"Length mismatch at {path}: {len(obj1)} != {len(obj2)}")
            if fail_fast:
                return False
            differences_found = True
        for i, (x, y) in enumerate(zip(obj1, obj2)):
            result = deep_compare(x, y, visited, f"{path}[{i}]", fail_fast)
            if not result:
                if fail_fast:
                    return False
                differences_found = True
        return not differences_found

    # Objects with __dict__
    if hasattr(obj1, "__dict__"):
        return deep_compare(vars(obj1), vars(obj2), visited, path + ".__dict__", fail_fast)

    # Objects with __slots__
    if hasattr(obj1, "__slots__"):
        for slot in obj1.__slots__:
            val1 = getattr(obj1, slot, None)
            val2 = getattr(obj2, slot, None)
            result = deep_compare(val1, val2, visited, f"{path}.{slot}", fail_fast)
            if not result:
                if fail_fast:
                    return False
                differences_found = True
        return not differences_found

    # Primitive value mismatch
    print(f"Value mismatch at {path}: {obj1!r} != {obj2!r}")
    return not fail_fast or False

    
def load_datamodule_from_config(config_path: str, overrides: Optional[list] = None) -> MoleculeImprovementLightningDataModule:
    """
    Load the datamodule using hydra configuration.
    
    Args:
        config_path: Path to the hydra config file
        overrides: List of hydra overrides
        
    Returns:
        Instantiated datamodule
    """
    # Get the config directory and name
    config_path_obj = Path(config_path)
    config_dir = config_path_obj.parent
    config_name = config_path_obj.stem
    print(f"config_dir: {config_dir}")
    print(f"config_name: {config_name}")
    
    # Change to the config directory to make paths relative
    original_cwd = Path.cwd()
    # os.chdir(config_dir)
    # new_cwd = Path.cwd()
    # print('changed to config_dir', config_dir, 'new_cwd', new_cwd)
    print('original_cwd', original_cwd)
    
    try:
        # Clear any existing Hydra instance before initializing
        if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        
        # Initialize hydra with relative path
        hydra.initialize(config_path=str(config_dir), version_base=None)

        print('hydra initialized')
        
        if overrides is None:
            overrides = []
        
        cfg = hydra.compose(config_name=str(config_name), overrides=overrides)
        logger.info("Configuration loaded:")
        logger.info(OmegaConf.to_yaml(cfg))
        
        # Instantiate setup (for seed, torch settings, etc.)
        if hasattr(cfg, 'setup'):
            hydra.utils.instantiate(cfg.setup)
        
        # Instantiate the datamodule
        logger.info("Instantiating datamodule...")
        datamodule = hydra.utils.instantiate(cfg.data)
        transform = hydra.utils.instantiate(cfg.data.transform_fn)
        
        return datamodule, cfg, transform
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def main():
    """Main function to load and setup the datamodule."""
    args = parse_args()
    
    try:
        # Load datamodule from config
        datamodule, cfg = load_datamodule_from_config(args.config_path, args.overrides)
        
        if args.dry_run:
            logger.info("Dry run mode - datamodule instantiated but not setup")
            logger.info(f"Datamodule type: {type(datamodule)}")
            logger.info(f"Datamodule config: {datamodule}")
            return
        
        # Prepare and setup the datamodule
        logger.info("Preparing data...")
        datamodule.prepare_data()
        
        logger.info(f"Setting up datamodule for stage: {args.stage}")
        datamodule.setup(stage=args.stage)
        
        # Print some information about the datasets
        logger.info("Datamodule setup complete!")
        logger.info(f"Training dataset size: {len(datamodule.train_dataset) if hasattr(datamodule, 'train_dataset') else 'N/A'}")
        logger.info(f"Validation dataset size: {len(datamodule.val_dataset) if hasattr(datamodule, 'val_dataset') else 'N/A'}")
        logger.info(f"Test dataset size: {len(datamodule.test_dataset) if hasattr(datamodule, 'test_dataset') else 'N/A'}")
        
        # Get a sample batch from training dataloader
        if args.stage == "fit" and hasattr(datamodule, 'train_dataloader'):
            logger.info("Getting a sample batch from training dataloader...")
            train_loader = datamodule.train_dataloader()
            sample_batch = next(iter(train_loader))
            logger.info(f"Sample batch keys: {sample_batch.keys()}")
            logger.info(f"Sample batch shapes: {[(k, v.shape) for k, v in sample_batch.items()]}")
        
        return datamodule
        
    except Exception as e:
        logger.error(f"Error loading datamodule: {e}")
        raise


if __name__ == "__main__":
    datamodule = main()
