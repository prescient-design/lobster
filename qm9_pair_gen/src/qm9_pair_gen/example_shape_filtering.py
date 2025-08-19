#!/usr/bin/env python3
"""
Example script demonstrating how to use shape Tanimoto filtering
in the MoleculeImprovementLightningDataModule.

This shows the clean, modular approach where:
1. Pair generation stores all shape Tanimoto distances (no filtering)
2. Data loading applies shape Tanimoto filtering per molecule using the utility function
3. No additional storage is needed for different filtering options
4. Supports both percentile and concrete number of pairs per molecule
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from lobster.data._molecule_improvement_datamodule import MoleculeImprovementLightningDataModule

def main():
    """Demonstrate shape Tanimoto filtering with both percentile and num_pairs options."""
    
    # Example 1: No shape filtering (original behavior)
    print("=== Example 1: No shape filtering ===")
    datamodule_no_shape = MoleculeImprovementLightningDataModule(
        root="/path/to/your/data",  # Replace with actual path
        utility_key="utility_0",
        delta=0.3,  # Morgan Tanimoto distance threshold
        epsilon=0.1,  # Utility improvement threshold
        shape_tanimoto_percentile=None,  # No shape filtering
        shape_tanimoto_num_pairs=None,  # No shape filtering
        batch_size=4,
    )
    
    # Example 2: Keep only closest 10% of pairs by shape Tanimoto distance per molecule
    print("=== Example 2: 10% shape filtering ===")
    datamodule_10_percent = MoleculeImprovementLightningDataModule(
        root="/path/to/your/data",  # Replace with actual path
        utility_key="utility_0",
        delta=0.3,  # Morgan Tanimoto distance threshold
        epsilon=0.1,  # Utility improvement threshold
        shape_tanimoto_percentile=10.0,  # Keep closest 10% by shape Tanimoto
        shape_tanimoto_num_pairs=None,  # Not used when percentile is specified
        batch_size=4,
    )
    
    # Example 3: Keep only closest 5 pairs by shape Tanimoto distance per molecule
    print("=== Example 3: 5 pairs per molecule shape filtering ===")
    datamodule_5_pairs = MoleculeImprovementLightningDataModule(
        root="/path/to/your/data",  # Replace with actual path
        utility_key="utility_0",
        delta=0.3,  # Morgan Tanimoto distance threshold
        epsilon=0.1,  # Utility improvement threshold
        shape_tanimoto_percentile=None,  # Not used when num_pairs is specified
        shape_tanimoto_num_pairs=5,  # Keep closest 5 pairs by shape Tanimoto
        batch_size=4,
    )
    
    # Example 4: Keep only closest 20 pairs by shape Tanimoto distance per molecule
    print("=== Example 4: 20 pairs per molecule shape filtering ===")
    datamodule_20_pairs = MoleculeImprovementLightningDataModule(
        root="/path/to/your/data",  # Replace with actual path
        utility_key="utility_0",
        delta=0.3,  # Morgan Tanimoto distance threshold
        epsilon=0.1,  # Utility improvement threshold
        shape_tanimoto_percentile=None,  # Not used when num_pairs is specified
        shape_tanimoto_num_pairs=20,  # Keep closest 20 pairs by shape Tanimoto
        batch_size=4,
    )
    
    print("\nKey benefits of this approach:")
    print("1. Single parquet file stores all shape Tanimoto distances")
    print("2. Flexible filtering at data loading time using utility function")
    print("3. No storage overhead for different filtering options")
    print("4. Clean separation of concerns: generation vs. filtering")
    print("5. Easy to experiment with different filtering strategies")
    print("6. Reuses existing utility function for consistency")
    print("7. Supports both percentile and concrete number filtering")

if __name__ == "__main__":
    main()
