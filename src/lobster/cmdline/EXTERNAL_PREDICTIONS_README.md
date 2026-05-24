# External Predictions Analysis Tool

A standalone tool to analyze external predicted protein structures against ground truth structures, using the same metrics (TM-score, RMSD) as the lobster generation pipeline.

## Features

- **Sequence Validation**: Automatically checks that predicted and ground truth sequences match
- **Consistent Metrics**: Uses the same TM-score and RMSD calculations as `distributed_generate.py`
- **Kabsch Alignment**: RMSD computed via `align_and_compute_rmsd` (Kabsch superposition)
- **Table Format**: Results formatted like `aggregate_results.py` for easy comparison
- **Comprehensive Statistics**: Mean, std, min, max, median, pass rates
- **CSV Output**: Both detailed and summary CSV files

## Quick Start

### For DPLM2 Folding Predictions

```bash
./analyze_dplm2_folding.sh
```

### Manual Usage

```bash
uv run python analyze_external_predictions.py \
    --pred-dir /path/to/predicted/pdbs \
    --gt-dir /path/to/ground_truth/pt_files \
    --output analysis_results.csv \
    --device cuda \
    --rmsd-threshold 2.0
```

## Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--pred-dir` | str | Yes | Directory containing predicted PDB files |
| `--gt-dir` | str | Yes | Directory containing ground truth .pt files |
| `--output` | str | No | Output CSV file path (default: `external_predictions_analysis.csv`) |
| `--device` | str | No | Device for computation: `cpu` or `cuda` (default: `cpu`) |
| `--rmsd-threshold` | float | No | RMSD threshold for pass rate (default: 2.0 Å) |
| `--skip-sequence-mismatch` | flag | No | Skip structures with sequence mismatches (default: analyze with warning) |

## Input Requirements

### Predicted Structures
- **Format**: PDB files (`.pdb`)
- **Location**: All in one directory
- **Naming**: Filenames should contain structure ID (e.g., `5S9R_pred.pdb`, `5S9R.pdb`)
- **Content**: Backbone atoms (N, CA, C) required

### Ground Truth Structures
- **Format**: PyTorch tensor files (`.pt`)
- **Location**: All in one directory  
- **Naming**: Must match predicted structure IDs (e.g., `5S9R.pt`)
- **Content**: Must contain:
  - `coords_res`: Backbone coordinates `(L, 3, 3)` for N, CA, C atoms
  - `seq`: Sequence tensor `(L,)` with amino acid indices
  - `mask`: (Optional) Mask tensor `(L,)`

## Output Files

### 1. Detailed Results CSV
Example: `dplm2_folding_analysis.csv`

| Structure_ID | Length | Seq_Identity | TM_Score | RMSD | Pred_File | GT_File |
|--------------|--------|--------------|----------|------|-----------|---------|
| 5S9R | 169 | 100.0 | 0.8542 | 1.23 | 5S9R_pred.pdb | 5S9R.pt |
| 5SBK | 370 | 98.5 | 0.7821 | 2.45 | 5SBK_pred.pdb | 5SBK.pt |

### 2. Summary CSV
Example: `dplm2_folding_analysis_summary.csv`

| Total_Structures | Avg_TM_Score | Std_TM_Score | Avg_RMSD | Std_RMSD | Structures_RMSD<2.0 | Pct_RMSD<2.0 |
|------------------|--------------|--------------|----------|----------|---------------------|--------------|
| 449 | 0.7234 | 0.1521 | 2.15 | 1.03 | 312 | 69.49 |

## Metrics Explanation

### TM-Score (Template Modeling Score)
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - ≥ 0.5: Generally indicates same fold
  - ≥ 0.6: High structural similarity
  - ≥ 0.8: Very similar structures
- **Calculation**: Uses `tmtools.tm_align` on CA atoms
- **Reference**: Zhang & Skolnick (2004)

### RMSD (Root Mean Square Deviation)
- **Unit**: Ångströms (Å)
- **Interpretation**:
  - < 1.0 Å: Excellent
  - < 2.0 Å: Good (typical threshold)
  - < 4.0 Å: Acceptable
  - > 4.0 Å: Poor
- **Calculation**: 
  1. Kabsch superposition on CA atoms
  2. RMSD computed on aligned CA atoms
  3. Uses `align_and_compute_rmsd` function

## Example Output

```
================================================================================
SUMMARY STATISTICS
================================================================================
Total structures analyzed: 449
Structures with ground truth: 449/449

Sequence Identity:
  Mean: 99.87%
  Min:  95.20%
  Max:  100.00%
  Exact matches: 442/449 (98.4%)

TM-Score:
  Mean: 0.7234
  Std:  0.1521
  Min:  0.3412
  Max:  0.9823
  Median: 0.7521

RMSD:
  Mean: 2.1543 Å
  Std:  1.0321 Å
  Min:  0.3421 Å
  Max:  5.2341 Å
  Median: 1.8932 Å

Structures with RMSD < 2.0 Å: 312/449 (69.5%)
```

## Structure Matching

The script automatically matches predicted and ground truth structures by extracting structure IDs from filenames:

- `5S9R_pred.pdb` → ID: `5S9R`
- `5S9R_folded.pdb` → ID: `5S9R`
- `5S9R.pdb` → ID: `5S9R`

All match with ground truth: `5S9R.pt`

## Comparison with Lobster Generation Pipeline

This tool uses **identical metrics** to the lobster generation pipeline:

| Component | Lobster Pipeline | This Tool | Match |
|-----------|------------------|-----------|-------|
| TM-score calculation | `tmtools.tm_align` | `tmtools.tm_align` | ✓ |
| RMSD calculation | `align_and_compute_rmsd` | `align_and_compute_rmsd` | ✓ |
| Kabsch alignment | `kabsch_torch_batched` | `kabsch_torch_batched` | ✓ |
| Atom selection | CA atoms | CA atoms | ✓ |
| Table format | `aggregate_results.py` | Similar format | ✓ |

## Dependencies

- `torch`
- `pandas`
- `numpy`
- `loguru`
- `tmtools`
- `lobster` (for PDB parsing and metric functions)

All dependencies are included in the lobster environment.

## Sequence Validation

The script automatically validates that predicted and ground truth sequences match:

### Perfect Match (100% identity)
```
✓ Sequences match perfectly - analysis proceeds normally
```

### Sequence Mismatch (< 100% identity)
```
WARNING: 5SBK: Sequence mismatch - 98.5% identity (365/370 residues match)
  First difference at position 42: predicted='ACDEFG' ground_truth='ACGEFG'
```

**Behavior:**
- **Default**: Analyzes structure anyway with warning
- **With `--skip-sequence-mismatch`**: Skips structure entirely

**Output:**
- `Seq_Identity` column shows percentage match
- Summary shows worst mismatches
- Overall sequence identity statistics

## Troubleshooting

### Sequence Mismatch
If sequences don't match between predicted and ground truth:
- **Cause**: Different folding task (e.g., inverse folding changed sequence)
- **Default**: Analysis continues with warning
- **Solution**: Use `--skip-sequence-mismatch` to exclude these structures
- **Note**: Metrics are still computed correctly on matching residue positions

### Length Mismatch
If predicted and ground truth structures have different lengths:
- Script automatically truncates to shorter length
- Warning message displayed
- Analysis continues with truncated structures

### Missing Ground Truth
If ground truth file not found:
- Warning message displayed
- Structure skipped
- Analysis continues with remaining structures

### PDB Loading Errors
If PDB file cannot be parsed:
- Error message displayed with details
- Structure skipped
- Uses lobster's `load_pdb` function (same as in generation pipeline)
- Check PDB file format and backbone atoms (N, CA, C required)

## Citation

If you use this tool, please cite the lobster paper and the TM-align paper:

```
Zhang, Y., & Skolnick, J. (2004). 
Scoring function for automated assessment of protein structure template quality. 
Proteins: Structure, Function, and Bioinformatics, 57(4), 702-710.
```

## Contact

For issues or questions about this tool, please contact the lobster development team.

