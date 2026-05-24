# External Analysis Tools

Quick reference for analyzing external predicted structures and extracting sequences from .pt files.

## 📁 Location
All tools are in: `src/lobster/cmdline/`

## 🔧 Tools

### 1. Analyze External Predictions
**File:** `analyze_external_predictions.py`

Compare predicted PDB structures against ground truth using TM-score and RMSD.

```bash
# Quick run (for DPLM2)
cd /homefs/home/lisanzas/scratch/Develop/lobster
./src/lobster/cmdline/analyze_dplm2_folding.sh

# Manual run
uv run python src/lobster/cmdline/analyze_external_predictions.py \
    --pred-dir /path/to/predicted/pdbs \
    --gt-dir /path/to/ground_truth/pt_files \
    --output results.csv \
    --device cuda
```

**See:** `EXTERNAL_PREDICTIONS_README.md` for full documentation

---

### 2. Extract Sequences to FASTA
**File:** `extract_pt_to_fasta.py`

Extract amino acid sequences from .pt files to FASTA format.

```bash
# Default (all sequences from test set)
uv run python src/lobster/cmdline/extract_pt_to_fasta.py

# Custom input/output
uv run python src/lobster/cmdline/extract_pt_to_fasta.py \
    --input-dir /data2/lisanzas/multi_flow_data/test_set_filtered_pt \
    --output sequences.fasta

# With options
uv run python src/lobster/cmdline/extract_pt_to_fasta.py --help
```

**Options:**
- `--input-dir`: Directory with .pt files (default: test_set_filtered_pt)
- `--output`: Output FASTA file (default: test_set_filtered_sequences.fasta)
- `--truncate-at-x`: Truncate sequences at first X (unknown residue)

---

## 📊 Quick Example Workflow

```bash
cd /homefs/home/lisanzas/scratch/Develop/lobster

# 1. Extract sequences for input
uv run python src/lobster/cmdline/extract_pt_to_fasta.py \
    --output my_sequences.fasta

# 2. Run prediction with external tool (e.g., DPLM2, ESMFold)
# ... (your prediction pipeline)

# 3. Analyze results
uv run python src/lobster/cmdline/analyze_external_predictions.py \
    --pred-dir /path/to/predictions/pdb/ \
    --gt-dir /data2/lisanzas/multi_flow_data/test_set_filtered_pt/ \
    --output analysis.csv \
    --device cuda

# Results saved to:
# - analysis.csv (detailed per-structure)
# - analysis_summary.csv (aggregate statistics)
```

## 🎯 Common Use Cases

### Compare Model A vs Model B
```bash
# Analyze Model A
uv run python src/lobster/cmdline/analyze_external_predictions.py \
    --pred-dir /path/to/modelA/pdbs --gt-dir /ground/truth --output modelA.csv

# Analyze Model B  
uv run python src/lobster/cmdline/analyze_external_predictions.py \
    --pred-dir /path/to/modelB/pdbs --gt-dir /ground/truth --output modelB.csv

# Compare results in Python/R using the CSV files
```

### Prepare Data for External Tools
```bash
# Extract clean sequences (no X residues)
uv run python src/lobster/cmdline/extract_pt_to_fasta.py \
    --truncate-at-x \
    --output clean_sequences.fasta

# Use with AlphaFold, ESMFold, DPLM, etc.
```

## 📝 Notes

- Both tools use the **same metrics** as lobster's generation pipeline
- TM-score calculated with `tmtools.tm_align`
- RMSD calculated with Kabsch alignment via `align_and_compute_rmsd`
- Sequence validation included (checks pred/GT sequences match)
- All dependencies included in lobster environment (`uv` managed)

## 🆘 Help

```bash
# Full documentation
cat src/lobster/cmdline/EXTERNAL_PREDICTIONS_README.md

# Command help
uv run python src/lobster/cmdline/analyze_external_predictions.py --help
uv run python src/lobster/cmdline/extract_pt_to_fasta.py --help
```

