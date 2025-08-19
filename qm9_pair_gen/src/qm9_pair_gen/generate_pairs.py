#!/usr/bin/env python3
"""
Main script for generating molecule pairs using Hydra configuration.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path

from pair_generation_src import PairGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_split_specific_parquet_files(config):
    try:
        import polars as pl
        
        # Read the main pairs file
        pairs_df = pl.read_parquet(Path(config.output_dir) / "pairs.parquet")
        
        # Create a separate file for each split
        for split in config.splits:
            split_df = pairs_df.filter(pl.col("split") == split)
            output_file = Path(config.output_dir) / f"pairs_{split}.parquet"
            split_df.write_parquet(output_file)
            logger.info(f"✅ Created {split} pairs file with {len(split_df)} rows: {output_file}")
            
    except Exception as e:
        logger.error(f"❌ Error creating split-specific files: {e}")
        raise e
    

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    """Main function to run the pair generation with Hydra configuration."""
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    
    # Create generator and run
    generator = PairGenerator(config)
    generator.generate_parquet_files()

    # Create split-specific parquet files
    logger.info("Creating split-specific parquet files...")
    
    create_split_specific_parquet_files(config)
    
    # The config file is automatically saved in the output directory by PairGenerator
    config_file_path = Path(config.output_dir) / "config.yaml"
    logger.info(f"✅ Configuration saved to: {config_file_path}")
    
    logger.info(f"\n✅ Parquet files generated successfully!")
    logger.info(f"📁 Output directory: {config.output_dir}")
    logger.info(f"🧪 Dataset: {config.dataset_name}")
    
    # Show which pair generation method is being used
    if config.percentile_threshold is not None:
        logger.info(f"🧪 Pair generation method: percentile_threshold = {config.percentile_threshold}%")
    else:
        logger.info(f"🧪 Pair generation method: max_pairs_per_mol = {config.max_pairs_per_mol}")
    
    logger.info(f"🧪 Split used: {config.get('split', 'train')}")
    logger.info(f"🧪 Test mode: {config.test_mode}")
    if config.test_mode:
        logger.info(f"🧪 Test sizes - Train: {config.test_sizes[0]}, Val: {config.test_sizes[1]}, Test: {config.test_sizes[2]}")
    logger.info(f"🧪 Molecules processed: {config.max_molecules}")
    logger.info(f"🧪 Log extra metrics: {config.log_extra_metrics}")
    logger.info(f"🧪 Convert to Parquet: {config.save_as_parquet}")
    logger.info(f"\n📊 You can monitor progress while running:")
    logger.info(f"python monitor_pairs.py '{config.output_dir}/pairs.arrow' 30")
    logger.info(f"\nYou can now use these files with lobster training:")
    logger.info(f"lobster_train data=molecule_improvement data.root='{config.output_dir}'")


if __name__ == "__main__":
    main() 