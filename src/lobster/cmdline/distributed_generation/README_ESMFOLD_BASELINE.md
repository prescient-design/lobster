# ESMFold Baseline for Forward Folding Comparison

This guide explains how to run ESMFold as a baseline for forward folding tasks and compare it to your genUME model results.

## Overview

The ESMFold baseline:
- Takes input structures and extracts sequences
- Predicts structures using **ESMFold only** (sequence-to-structure prediction)
- Compares predictions to ground truth structures
- Outputs the **same CSV format** as forward_folding mode
- Can be aggregated using the **existing aggregation script**

This allows for direct comparison between:
- **ESMFold baseline**: Sequence → Structure (ESMFold prediction)
- **genUME forward folding**: Sequence + Structure → New Structure (your model)

## Quick Start

### Single-Run Mode (For Testing/Small Datasets)

```bash
# Run ESMFold baseline on all structures
uv run python -m lobster.cmdline.esmfold_baseline \
    --config-path "../hydra_config/experiment" \
    --config-name esmfold_baseline
```

The config file is at `src/lobster/hydra_config/experiment/esmfold_baseline.yaml`:
```yaml
output_dir: "./examples/esmfold_baseline"
seed: 12345
generation:
  input_structures: "/data2/lisanzas/multi_flow_data/test_set_filtered_pt/*.pt"
  batch_size: 5
  max_length: 512
```

### Distributed Mode (For Large Datasets)

For 449 structures, distribute across 90 jobs (5 structures per job):

#### Step 1: Initialize WandB Sweep

```bash
wandb sweep src/lobster/cmdline/distributed_generation/wandb_config_esmfold_baseline.yaml
```

This will output a sweep ID like: `prescient-design/lobster-esmfold-baseline/abc123xyz`

#### Step 2: Create SLURM Submit Script

Create `submit_esmfold_baseline.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=esmfold_baseline
#SBATCH --array=0-89                    # 90 jobs total
#SBATCH --output=logs/esmfold_%A_%a.out
#SBATCH --error=logs/esmfold_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load environment
source activate lobster  # or your environment setup

# Set sweep ID (from Step 1)
SWEEP_ID="prescient-design/lobster-esmfold-baseline/abc123xyz"

# Run wandb agent
wandb agent $SWEEP_ID
```

#### Step 3: Submit Jobs

```bash
mkdir -p logs
sbatch submit_esmfold_baseline.sh
```

#### Step 4: Monitor Progress

```bash
# Check SLURM jobs
squeue -u $USER

# Check WandB dashboard
# Visit: https://wandb.ai/prescient-design/lobster-esmfold-baseline

# Check output directories
ls -la examples/esmfold_baseline/job_*/
```

#### Step 5: Aggregate Results

Once all jobs complete, aggregate results using the existing script:

```bash
uv run python src/lobster/cmdline/distributed_generation/aggregate_results.py \
    ./examples/esmfold_baseline \
    90 \
    --mode forward_folding \
    --no-foldseek
```

**Note**: Use `--mode forward_folding` since ESMFold baseline outputs the same CSV format. Use `--no-foldseek` because diversity analysis doesn't apply to baselines.

## Output Files

### Per-Job Outputs

Each `job_N` directory contains:

```
examples/esmfold_baseline/job_0/
├── esmfold_baseline_metrics_TIMESTAMP.csv          # Metrics CSV
├── sequences_esmfold_baseline_TIMESTAMP.csv        # Sequences CSV
├── esmfold_baseline_<name>_predicted.pdb           # ESMFold predictions
└── esmfold_baseline_<name>_ground_truth.pdb        # Ground truth structures
```

### Metrics CSV Format

Compatible with forward_folding mode:

```csv
run_id,timestamp,mode,plddt,tm_score,rmsd,sequence_length,input_file
esmfold_baseline_batch_000_0,2025-01-13 10:30:00,esmfold_baseline,85.23,0.85,1.23,150,structure_001
esmfold_baseline_batch_000_1,2025-01-13 10:31:00,esmfold_baseline,82.45,0.78,1.89,200,structure_002
...
```

### Aggregated Results

After running `aggregate_results.py`:

```
examples/esmfold_baseline/aggregated/
├── combined_forward_folding_metrics.csv      # All metrics combined
├── combined_forward_folding_sequences.csv    # All sequences combined
├── summary_per_structure.csv                 # Per-structure summary
├── overall_summary.csv                       # Overall statistics
└── *.pdb                                     # All PDB files copied here
```

## Comparing to Forward Folding Results

### Run Both Methods

```bash
# 1. Run genUME forward folding (your method)
wandb sweep src/lobster/cmdline/distributed_generation/wandb_config_forward_folding.yaml
# ... submit jobs and aggregate ...

# 2. Run ESMFold baseline (comparison)
wandb sweep src/lobster/cmdline/distributed_generation/wandb_config_esmfold_baseline.yaml
# ... submit jobs and aggregate ...
```

### Compare Results

```python
import pandas as pd

# Load aggregated results
esmfold_results = pd.read_csv("examples/esmfold_baseline/aggregated/overall_summary.csv")
forward_folding_results = pd.read_csv("examples/generated_forward_folding/aggregated/overall_summary.csv")

# Compare metrics
print("ESMFold Baseline:")
print(f"  Avg TM-Score: {esmfold_results['Avg_TM_Score'].values[0]:.3f}")
print(f"  Avg RMSD: {esmfold_results['Avg_RMSD'].values[0]:.2f} Å")
print(f"  Avg pLDDT: {esmfold_results['Avg_pLDDT'].values[0]:.2f}")

print("\ngenUME Forward Folding:")
print(f"  Avg TM-Score: {forward_folding_results['Avg_TM_Score'].values[0]:.3f}")
print(f"  Avg RMSD: {forward_folding_results['Avg_RMSD'].values[0]:.2f} Å")

# Calculate improvement
tm_improvement = (forward_folding_results['Avg_TM_Score'].values[0] - 
                  esmfold_results['Avg_TM_Score'].values[0])
print(f"\nTM-Score Improvement: {tm_improvement:+.3f}")
```

## Configuration Options

### Adjusting Batch Size

For memory constraints, adjust batch size in the config:

```yaml
generation:
  batch_size: 1  # Process 1 structure at a time (lower memory)
  # or
  batch_size: 10  # Process 10 structures at once (faster, more memory)
```

### Adjusting Job Distribution

For 449 structures:
- **5 structures/job = 90 jobs** (default, good for most SLURM clusters)
- **10 structures/job = 45 jobs** (fewer jobs, longer runtime per job)
- **1 structure/job = 449 jobs** (maximum parallelism)

Update in `wandb_config_esmfold_baseline.yaml`:
```yaml
parameters:
  structures_per_job:
    value: 10  # Adjust this
  total_structures:
    value: 449
  job_id:
    values: [0, 1, 2, ..., 44]  # Adjust range accordingly
```

## Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size
generation:
  batch_size: 1

# Or request more memory in SLURM
#SBATCH --mem=64G
```

### ESMFold Model Not Found

```bash
# Ensure ESMFold is properly installed
uv run python -c "from lobster.model._lobster_fold import LobsterPLMFold; print('OK')"
```

### Structure Loading Errors

Check that .pt files have the correct format:
```python
import torch
data = torch.load("structure.pt", weights_only=False)
print(data.keys())  # Should have: sequence, coords_res, mask, indices
```

### WandB Sweep Not Starting

```bash
# Check sweep status
wandb sweep --help

# Verify config file
cat src/lobster/cmdline/distributed_generation/wandb_config_esmfold_baseline.yaml
```

## Expected Performance

Based on ESMFold benchmarks:
- **TM-Score**: 0.70-0.85 (varies by structure difficulty)
- **RMSD**: 1.5-3.0 Å (for well-folded proteins)
- **pLDDT**: 75-90 (confidence score)

Your genUME forward folding should ideally **exceed these baseline metrics** by leveraging structural information.

## Advanced Usage

### Custom Input Structures

```yaml
generation:
  # Single file
  input_structures: "/path/to/structure.pdb"
  
  # Directory
  input_structures: "/path/to/structures/"
  
  # Glob pattern
  input_structures: "/path/to/structures/*.pt"
  
  # List of files
  input_structures:
    - "/path/to/structure1.pdb"
    - "/path/to/structure2.pdb"
    - "/path/to/structure3.pt"
```

### Running on Specific Structures

```bash
# Create a subset config
cat > subset_esmfold_baseline.yaml << EOF
output_dir: "./examples/esmfold_baseline_subset"
seed: 12345
generation:
  input_structures:
    - "/data2/lisanzas/multi_flow_data/test_set_filtered_pt/structure1.pt"
    - "/data2/lisanzas/multi_flow_data/test_set_filtered_pt/structure2.pt"
  batch_size: 2
  max_length: 512
EOF

# Run
uv run python -m lobster.cmdline.esmfold_baseline \
    --config-path "." \
    --config-name subset_esmfold_baseline
```

## Integration with Existing Workflows

The ESMFold baseline is designed to integrate seamlessly:

1. **Same CSV format** as forward_folding mode
2. **Same aggregation script** (aggregate_results.py)
3. **Same metrics columns**: tm_score, rmsd, sequence_length, input_file
4. **Compatible with existing analysis pipelines**

This allows you to:
- Use the same plotting scripts
- Use the same statistical analysis
- Direct comparison in the same tables/figures
- No code changes needed for downstream analysis

## Questions?

For issues or questions:
1. Check the logs: `logs/esmfold_*.err`
2. Verify CSV output format matches forward_folding
3. Ensure all dependencies are installed
4. Check WandB dashboard for job status

