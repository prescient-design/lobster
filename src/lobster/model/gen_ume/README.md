# Gen-UME: Generative Unified Molecular Encoder

Gen-UME is a discrete diffusion-based generative model for protein structure and sequence design. It supports three generation modes: **unconditional generation**, **inverse folding**, and **forward folding**.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Generation Modes](#generation-modes)
  - [Unconditional Generation](#1-unconditional-generation)
  - [Inverse Folding](#2-inverse-folding)
  - [Forward Folding](#3-forward-folding)
- [Benchmark Results](#benchmark-results)
- [Key Parameters](#key-parameters)
- [Advanced Features](#advanced-features)
- [Tips and Best Practices](#tips-and-best-practices)

## Overview

Gen-UME generates protein structures and sequences using a unified diffusion model that operates on both modalities simultaneously. The model can:

- **Generate novel proteins** from scratch (unconditional)
- **Design sequences** for given structures (inverse folding)
- **Predict structures** from sequences (forward folding)

### Model Checkpoint

The default checkpoint is hosted on S3:
```
s3://prescient-lobster/ume/gen_ume/checkpoints/gen-ume-small-PDB-90M.ckpt
```

## Installation

Ensure you have the lobster package installed:

```bash
cd /path/to/lobster
uv pip install -e .
```

## Generation Modes

### 1. Unconditional Generation

Generate novel protein structures and sequences from scratch.

#### Basic Usage

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional
```

#### Configuration

Create a config file (e.g., `my_unconditional.yaml`):

```yaml
# Output directory
output_dir: "./examples/my_generation"

# Random seed for reproducibility
seed: 12345

# Model configuration
model:
  _target_: lobster.model.gen_ume.UMESequenceStructureEncoderLightningModule
  ckpt_path: "s3://prescient-lobster/ume/gen_ume/checkpoints/gen-ume-small-PDB-90M.ckpt"

# Generation parameters
generation:
  mode: unconditional
  length: [100, 200, 300, 400, 500]  # Sequence lengths to generate
  num_samples: 10                     # Samples per length
  nsteps: 1000                        # Diffusion steps
  batch_size: 1
  
  # Temperature and stochasticity control
  temperature_seq: 0.458
  temperature_struc: 0.358
  stochasticity_seq: 30
  stochasticity_struc: 70
  
  # ESMFold validation
  use_esmfold: true
  max_length: 512
  
  # Metrics and visualization
  save_csv_metrics: true
  create_plots: true
```

Then run:

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "/path/to/config" \
    --config-name my_unconditional
```

#### Self-Reflection (Recommended)

Enable self-reflection to improve structure-sequence consistency:

```yaml
generation:
  enable_self_reflection: true
  
  self_reflection:
    use_esmfold_validation: false  # Enable for detailed metrics
    
    forward_folding:
      nsteps: 100
      temperature_seq: 0.297
      temperature_struc: 0.110
      stochasticity_seq: 10
      stochasticity_struc: 30
    
    inverse_folding:
      nsteps: 200
      temperature_seq: 0.164
      temperature_struc: 1.0
      stochasticity_seq: 20
      stochasticity_struc: 10
    
    quality_control:
      enable_tm_threshold: true
      min_tm_score_forward: 0.833
      enable_min_percent_identity_threshold: true
      min_percent_identity: 50
      enable_max_percent_identity_threshold: true
      max_percent_identity: 100
      max_retries: 30
```

### 2. Inverse Folding

Generate sequences for given protein structures (sequence design).

#### Basic Usage

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_inverse_folding \
    generation.input_structures="path/to/structures/*.pdb"
```

#### Configuration

Create a config file (e.g., `my_inverse_folding.yaml`):

```yaml
# Output directory
output_dir: "./examples/my_inverse_folding"

# Random seed
seed: 54321

# Model configuration
model:
  _target_: lobster.model.gen_ume.UMESequenceStructureEncoderLightningModule
  ckpt_path: "s3://prescient-lobster/ume/gen_ume/checkpoints/gen-ume-small-PDB-90M.ckpt"

# Generation settings
generation:
  mode: inverse_folding
  nsteps: 200
  batch_size: 1
  n_trials: 3  # Generate multiple designs and select best
  
  # Temperature parameters for inverse folding
  temperature_seq: 0.164
  temperature_struc: 1.0
  stochasticity_seq: 20
  stochasticity_struc: 10
  
  n_designs_per_structure: 10  # Number of sequences per structure
  
  # Input structures (multiple formats supported)
  # Single file:
  input_structures: "/path/to/structure.pdb"
  
  # Or directory:
  # input_structures: "/path/to/pdb/directory/"
  
  # Or glob pattern:
  # input_structures: "/path/to/structures/*.pdb"
  
  # Or list of files:
  # input_structures: 
  #   - "/path/to/file1.pdb"
  #   - "/path/to/file2.pdb"
  
  # ESMFold validation (recommended)
  use_esmfold: true
  max_length: 512
```

#### Multi-Chain Support

For multi-chain structures, specify which chains to predict:

```yaml
generation:
  esmfold_chain_groups:
    - [A, B]      # Predict chains A and B together
    - [C]         # Predict chain C separately
```

If not specified, all chains will be predicted together.

### 3. Forward Folding

Generate structures from sequences (structure prediction).

#### Basic Usage

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_forward_folding \
    generation.input_structures="path/to/structures/*.pdb"
```

**Note:** Despite the name `input_structures`, forward folding extracts sequences from these structures to generate new structures.

#### Configuration

Create a config file (e.g., `my_forward_folding.yaml`):

```yaml
# Output directory
output_dir: "./examples/my_forward_folding"

# Random seed
seed: 54321

# Model configuration
model:
  _target_: lobster.model.gen_ume.UMESequenceStructureEncoderLightningModule
  ckpt_path: "s3://prescient-lobster/ume/gen_ume/checkpoints/gen-ume-small-PDB-90M.ckpt"

# Generation settings
generation:
  mode: forward_folding
  nsteps: 100
  batch_size: 1
  n_trials: 1
  
  # Temperature parameters for forward folding
  temperature_seq: 0.297
  temperature_struc: 0.110
  stochasticity_seq: 10
  stochasticity_struc: 30
  
  # Input structures (sequences extracted from these)
  input_structures: "/path/to/structures/*.pdb"
  
  max_length: 512
```

## Benchmark Results

Results from large-scale unconditional generation with self-reflection (100 samples per length):

| Length | Total Structures | RMSD<2.0 | % Pass | Clusters | Diversity % | Avg TM | Avg RMSD | Avg pLDDT |
|--------|-----------------|----------|--------|----------|-------------|--------|----------|-----------|
| 100    | 100             | 85       | 85.0%  | 25       | 25.0%       | 0.8203 | 1.963    | 0.7111    |
| 200    | 100             | 63       | 63.0%  | 19       | 19.0%       | 0.8043 | 2.467    | 0.6639    |
| 300    | 100             | 62       | 62.0%  | 23       | 23.0%       | 0.8447 | 2.015    | 0.6851    |
| 400    | 100             | 56       | 56.0%  | 13       | 13.0%       | 0.8505 | 2.191    | 0.7177    |
| 500    | 91              | 31       | 34.1%  | 10       | 11.0%       | 0.8344 | 2.730    | 0.7283    |

**Metrics Explanation:**
- **RMSD<2.0**: Number of structures with RMSD < 2.0 Å between gen-UME and ESMFold predictions
- **% Pass**: Percentage of structures passing RMSD threshold
- **Clusters**: Number of unique structural clusters (Foldseek, TM-score threshold 0.5)
- **Diversity %**: Percentage of unique structures (clusters/total)
- **Avg TM**: Average TM-score between gen-UME structure and ESMFold prediction
- **Avg RMSD**: Average RMSD between gen-UME structure and ESMFold prediction  
- **Avg pLDDT**: Average pLDDT (confidence score) from ESMFold prediction

**Key Observations:**
- Shorter sequences (100-200 AA) show better consistency with ESMFold
- Self-reflection improves structure quality across all lengths
- Diversity remains high (10-25%) indicating generation of distinct structures
- High TM-scores (>0.8) indicate good structural quality

## Key Parameters

### Temperature and Stochasticity

Control the randomness and exploration of the generation process:

| Parameter | Range | Effect | Recommended Values |
|-----------|-------|--------|-------------------|
| `temperature_seq` | 0.1-1.0 | Sequence randomness | Unconditional: 0.45, Inverse: 0.16, Forward: 0.30 |
| `temperature_struc` | 0.1-1.0 | Structure randomness | Unconditional: 0.35, Inverse: 1.0, Forward: 0.11 |
| `stochasticity_seq` | 0-100 | Sequence noise steps | Unconditional: 30, Inverse: 20, Forward: 10 |
| `stochasticity_struc` | 0-100 | Structure noise steps | Unconditional: 70, Inverse: 10, Forward: 30 |

**Tips:**
- **Lower temperature** = more deterministic, conservative outputs
- **Higher temperature** = more diverse, exploratory outputs
- **Higher stochasticity** = more diffusion steps with noise injection

### Generation Steps

| Mode | Recommended nsteps | Notes |
|------|-------------------|-------|
| Unconditional | 1000 | Higher steps for de novo generation |
| Unconditional + Self-Reflection | Forward: 100, Inverse: 200 | Refinement needs fewer steps |
| Inverse Folding | 200 | Structure constrains generation |
| Forward Folding | 100 | Sequence constrains generation |

## Advanced Features

### Distributed Generation

For large-scale generation, use the distributed generation system with WandB:

```bash
# See distributed generation README
cd src/lobster/cmdline/distributed_generation
python create_job_config.py --total_samples 100 --samples_per_job 5
```

See [Distributed Generation README](../../cmdline/distributed_generation/README.md) for details.

### Foldseek Diversity Analysis

Automatically cluster generated structures by structural similarity:

```yaml
generation:
  calculate_foldseek_diversity: true
  foldseek_bin_path: "/path/to/foldseek/bin"
  foldseek_tmscore_threshold: 0.5  # TM-score cutoff for clustering
  rmsd_threshold_for_diversity: 2.0  # Only cluster high-quality structures
```

### Asynchronous Sampling

Enable asynchronous sequence and structure sampling for faster generation:

```yaml
generation:
  asynchronous_sampling: true  # Default: false
```

**Note:** This can significantly speed up generation but may affect reproducibility.

## Tips and Best Practices

### 1. Start Small
Begin with small test runs to validate configurations:
```yaml
length: [100]
num_samples: 2
nsteps: 100  # Reduced for testing
```

### 2. Use Self-Reflection for Quality
For unconditional generation, always enable self-reflection to improve ESMFold metrics:
```yaml
enable_self_reflection: true
```

### 3. Enable ESMFold Validation
ESMFold provides crucial quality metrics:
```yaml
use_esmfold: true
max_length: 512  # Adjust based on your sequences
```

### 4. Batch Size Selection
- **GPU Memory Limited**: Use `batch_size: 1`
- **Long sequences (>400)**: Use `batch_size: 1`
- **Short sequences (<200)**: Can use `batch_size: 2-4`

### 5. Output Organization
```yaml
output_dir: "./examples/generation_YYYYMMDD_description"
```

Always use descriptive output directories with dates for tracking experiments.

### 6. Reproducibility
```yaml
seed: 12345  # Set seed for reproducible results
```

### 7. Monitor Progress
```yaml
save_csv_metrics: true
create_plots: true
```

Enables CSV logging and automatic plotting of metrics.

### 8. Multi-Chain Design
For inverse folding of multi-chain complexes:
```yaml
esmfold_chain_groups:
  - [A, B]  # Design interface
  - [C]     # Design separately
```

### 9. Quality Control
Use quality control thresholds to filter poor designs:
```yaml
self_reflection:
  quality_control:
    min_tm_score_forward: 0.833
    min_percent_identity: 50
    max_percent_identity: 100
    max_retries: 30
```

### 10. Structure File Formats
Supported input formats:
- **PDB files** (`.pdb`)
- **mmCIF files** (`.cif`)
- **PyTorch tensors** (`.pt`)

## Output Files

After generation, you'll find:

```
output_dir/
├── generated_structure_length_XXX_YYY.pdb     # Generated structures
├── generated_structure_length_XXX_YYY_esmfold_000.pdb  # ESMFold predictions
├── unconditional_metrics_TIMESTAMP.csv        # Metrics for all samples
├── unconditional_sequences_TIMESTAMP.csv      # Generated sequences
├── unconditional_combined_boxplots_TIMESTAMP.png  # Visualizations
└── foldseek_results/                          # Diversity analysis (if enabled)
    └── length_XXX/
        ├── res_rep_seq.fasta                  # Cluster representatives
        └── res_cluster.tsv                    # Cluster assignments
```

## Citation

If you use Gen-UME in your research, please cite:

```
[Citation to be added]
```

## Support

For issues and questions:
- **GitHub Issues**: [prescient-design/lobster](https://github.com/prescient-design/lobster)
- **Documentation**: See `src/lobster/cmdline/generate.py` for implementation details
- **Examples**: Check `src/lobster/hydra_config/experiment/` for example configurations

---

**Last Updated**: November 2025

