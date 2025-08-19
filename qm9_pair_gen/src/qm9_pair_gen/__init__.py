"""
QM9 Pair Generation utilities for molecular similarity and distance calculations.

This package provides utilities for:
- Computing Tanimoto distances between molecules
- Computing shape Tanimoto distances between molecules
- Filtering molecule pairs by distance metrics
- Visualizing molecular structures
- Generating molecule pairs for training
"""

from .utils_mol import (
    get_tanimoto_distance,
    get_shape_tanimoto_distance,
    filter_top_pairs_per_molecule,
    get_shape_tanimoto,
    visualize_mol_grid,
    is_valency_ok,
    human_readable_size,
)

__version__ = "0.1.0"

__all__ = [
    "get_tanimoto_distance",
    "get_shape_tanimoto_distance", 
    "filter_top_pairs_per_molecule",
    "get_shape_tanimoto",
    "visualize_mol_grid",
    "is_valency_ok",
    "human_readable_size",
]
