# QM9 Pair Generation

A Python package for QM9 pair generation and molecular distance calculations.

## Installation

Install in editable mode for development:

```bash
cd qm9_pair_gen
pip install -e .
```

Or use the provided installation script:

```bash
chmod +x install_qm9_pair_gen.sh
./install_qm9_pair_gen.sh
```

## Usage

### Basic Distance Calculations

```python
from qm9_pair_gen import get_tanimoto_distance, get_shape_tanimoto_distance

# Compute Tanimoto distance using Morgan fingerprints
smiles1 = "CCO"  # Ethanol
smiles2 = "CCCO"  # Propanol

tanimoto_dist = get_tanimoto_distance(smiles1, smiles2)
print(f"Tanimoto distance: {tanimoto_dist:.3f}")

# Compute shape Tanimoto distance (requires 3D conformers)
shape_dist = get_shape_tanimoto_distance(smiles1, smiles2, verbose=False)
print(f"Shape Tanimoto distance: {shape_dist:.3f}")
```

### Filtering Molecule Pairs

```python
from qm9_pair_gen import filter_top_pairs_per_molecule
import pandas as pd

# Filter pairs by percentile
df = pd.read_parquet("pairs.parquet")
filtered_df = filter_top_pairs_per_molecule(
    df, 
    property_key='shape_tanimoto_distance',
    percentile=10.0  # Keep closest 10%
)

# Filter pairs by number
filtered_df = filter_top_pairs_per_molecule(
    df, 
    property_key='shape_tanimoto_distance',
    num_pairs=5  # Keep closest 5 pairs per molecule
)
```

### Visualization

```python
from qm9_pair_gen import visualize_mol_grid
from rdkit import Chem

# Visualize a grid of molecules
mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCCO")]
visualize_mol_grid(mols, titles=["Ethanol", "Propanol"])
```

## Available Functions

- `get_tanimoto_distance(smiles1, smiles2)` - Compute Tanimoto distance using Morgan fingerprints
- `get_shape_tanimoto_distance(smiles1, smiles2, verbose=False)` - Compute shape Tanimoto distance
- `filter_top_pairs_per_molecule(df, property_key, percentile=None, num_pairs=None)` - Filter molecule pairs
- `get_shape_tanimoto(molA, molB, return_extra=True, verbose=True)` - Low-level shape Tanimoto calculation
- `visualize_mol_grid(mols, titles=None, ...)` - Visualize molecules in 3D
- `is_valency_ok(mol)` - Check if molecule has valid valencies
- `human_readable_size(size_bytes)` - Convert bytes to human-readable format

## Dependencies

- rdkit
- numpy
- pandas
- polars
- pyarrow
- py3Dmol
- tqdm
- atomic-datasets
- selfies

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

MIT License 