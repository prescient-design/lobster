# Gen-UME: Generative Unified Molecular Encoder

Gen-UME is a generative model for protein structure and sequence design based on discrete flow matching. It supports three generation modes: **unconditional generation**, **inverse folding**, and **forward folding**.

## Quick Start

```bash
# Unconditional: Generate novel proteins from scratch
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional

# Inverse Folding: Design sequences for structures
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_inverse_folding \
    generation.input_structures="path/to/structures/*.pdb"

# Forward Folding: Predict structures from sequences
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_forward_folding \
    generation.input_structures="path/to/structures/*.pdb"
```

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Model Checkpoints](#model-checkpoints)
- [Installation](#installation)
- [Generation Modes](#generation-modes)
  - [Unconditional Generation](#1-unconditional-generation)
  - [Inverse Folding](#2-inverse-folding)
  - [Forward Folding](#3-forward-folding)
- [Benchmark Results](#benchmark-results)
- [Key Parameters](#key-parameters)
- [Advanced Features](#advanced-features)
- [Tips and Best Practices](#tips-and-best-practices)
- [Protein-Ligand](#protein-ligand)
  - [Protein-Ligand Inverse Folding](#protein-ligand-inverse-folding)
  - [Protein-Ligand Forward Folding](#protein-ligand-forward-folding)

## Overview

Gen-UME generates protein structures and sequences using discrete flow matching, a unified generative modeling approach that operates on both modalities simultaneously. The model can:

- **Generate novel proteins** from scratch (unconditional)
- **Design sequences** for given structures (inverse folding)
- **Predict structures** from sequences (forward folding)

### Technical Approach

Gen-UME employs **discrete flow matching**, which models the generation process as a continuous-time flow on discrete state spaces (sequences) and continuous state spaces (structures). The model uses **tokenized structure representations** to encode protein backbone geometry, enabling efficient joint generation of sequence and structure.

## Model Checkpoints

For a complete list of all available checkpoints with detailed descriptions, see **[CHECKPOINTS.md](./CHECKPOINTS.md)**.

### Quick Reference

| Model | Size | S3 Path | Description |
|-------|------|---------|-------------|
| **Gen-UME 90M** | 1.1 GiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_90M_PDB.ckpt` | Smallest model, good for testing |
| **Gen-UME 450M** | 5.3 GiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_450M_2025-11-07_*.ckpt` | Medium model, balanced performance |
| **Gen-UME 750M** | 8.3 GiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_750M_2025-11-17_*.ckpt` | **Primary production model** |
| **Gen-UME 750M ESM Atlas** | 8.3 GiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_750M_ESM_Atlas_2026-01-04_*.ckpt` | Extended training data |

### Download Checkpoints

```bash
# Download Gen-UME 750M (recommended)
aws s3 cp s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/gen_ume_750M_2025-11-17_last.ckpt ./

# Download all Gen-UME checkpoints
aws s3 sync s3://prescient-pcluster-data/gen_ume/checkpoints/gen_ume/ ./checkpoints/
```

### Latent Generator Checkpoints

The Latent Generator provides the structure tokenization backbone:

| Model | Codebook | Size | S3 Path |
|-------|----------|------|---------|
| **LG PL FSQ 4375** | 4375 tokens (FSQ) | 295.8 MiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_fsq_4375_2026-01-05.ckpt` |
| **LG PL FSQ 4375/15360** | 4375/15360 tokens (asymmetric FSQ) | 360.2 MiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_fsq_4375_15360_2026-01-07.ckpt` |
| **LG PL 4096** | 4096 tokens (SLQ) | 292.9 MiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Protein_Ligand_4096_2026-01-05.ckpt` |
| **LG Ligand** | 512 tokens (SLQ) | 250.5 MiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_Ligand_2025-11-09.ckpt` |
| **LG Full Attention 2** | 256 tokens (SLQ) | 245.3 MiB | `s3://prescient-pcluster-data/gen_ume/checkpoints/latent_generator/LG_full_attention_2_2025-11-06.ckpt` |

**Additional checkpoints on HuggingFace:** LG Ligand 20A, LG 20A seq Aux, LG 20A seq 3di c6d Aux, LG 20A seq 3di c6d Aux PDB Pinder, and more. See [CHECKPOINTS.md](./CHECKPOINTS.md) for the complete list.

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

Use the provided configuration file:

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional
```

#### Configuration File

The example configuration is located at `src/lobster/hydra_config/experiment/generate_unconditional.yaml`:

```yaml
# Generation parameters
generation:
  mode: unconditional
  length: [100, 200, 300, 400, 500]  # Sequence lengths to generate
  num_samples: 10                     # Samples per length
  nsteps: 1000                        # Diffusion steps
  batch_size: 1
  
  # Temperature and stochasticity control
  temperature_seq: 0.4579796403264936
  temperature_struc: 0.35751879409731435
  stochasticity_seq: 30
  stochasticity_struc: 70
  
  # ESMFold validation
  use_esmfold: true
  max_length: 512
```

#### Override Parameters

You can override any parameter from the command line:

```bash
# Change output directory
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional \
    output_dir="./my_generation"

# Generate different lengths
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional \
    generation.length="[50,100,150]" \
    generation.num_samples=20
```

#### Self-Reflection (Enabled by Default)

The provided config already has self-reflection enabled to improve structure-sequence consistency. The self-reflection pipeline refines unconditionally generated structures through forward and inverse folding steps:

```yaml
generation:
  enable_self_reflection: true  # Already enabled in default config
  
  self_reflection:
    forward_folding:
      nsteps: 100
      temperature_seq: 0.2967457760634187
      temperature_struc: 0.1102551183666233
      stochasticity_seq: 10
      stochasticity_struc: 30
    
    inverse_folding:
      nsteps: 200
      temperature_seq: 0.16423763902324678
      temperature_struc: 1.0
      stochasticity_seq: 20
      stochasticity_struc: 10
    
    quality_control:
      enable_tm_threshold: true
      min_tm_score_forward: 0.8334123066155882
      min_percent_identity: 50
      max_percent_identity: 100
      max_retries: 30
```

To disable self-reflection:

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional \
    generation.enable_self_reflection=false
```

### 2. Inverse Folding

Generate sequences for given protein structures (sequence design).

#### Basic Usage

Use the provided configuration file and specify your input structures:

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_inverse_folding \
    generation.input_structures="path/to/structures/*.pdb"
```

#### Configuration File

The example configuration is located at `src/lobster/hydra_config/experiment/generate_inverse_folding.yaml`:

```yaml
# Generation settings
generation:
  mode: inverse_folding
  nsteps: 200
  batch_size: 1
  n_trials: 3  # Generate multiple designs and select best
  
  # Temperature parameters (optimized for inverse folding)
  temperature_seq: 0.16423763902324678
  temperature_struc: 1.0
  stochasticity_seq: 20
  stochasticity_struc: 10
  
  n_designs_per_structure: 10  # Number of sequences per structure
  
  # Input structures - update via command line or edit config
  input_structures: "test_data/inv_folding/9jl9.pdb"
  
  # ESMFold validation (recommended)
  use_esmfold: true
  max_length: 512
```

#### Input Structure Formats

Multiple input formats are supported:

```bash
# Single file
generation.input_structures="/path/to/structure.pdb"

# Directory (finds all PDB/CIF files)
generation.input_structures="/path/to/pdb/directory/"

# Glob pattern
generation.input_structures="/path/to/structures/*.pdb"

# Multiple files (use quotes)
generation.input_structures="[/path/to/file1.pdb,/path/to/file2.pdb]"
```

#### Multi-Chain Support

For multi-chain structures, specify which chains to predict:

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_inverse_folding \
    generation.input_structures="path/to/complex.pdb" \
    generation.esmfold_chain_groups="[[A,B],[C]]"
```

If not specified, all chains will be predicted together.

### 3. Forward Folding

Generate structures from sequences (structure prediction).

#### Basic Usage

Use the provided configuration file and specify your input structures:

```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_forward_folding \
    generation.input_structures="path/to/structures/*.pdb"
```

**Note:** Despite the name `input_structures`, forward folding extracts sequences from these structures to generate new structures.

#### Configuration File

The example configuration is located at `src/lobster/hydra_config/experiment/generate_forward_folding.yaml`:

```yaml
# Generation settings
generation:
  mode: forward_folding
  nsteps: 100
  batch_size: 1
  n_trials: 1
  
  # Temperature parameters (optimized for forward folding)
  temperature_seq: 0.2967457760634187
  temperature_struc: 0.1102551183666233
  stochasticity_seq: 10
  stochasticity_struc: 30
  
  # Input structures - sequences will be extracted from these
  input_structures: "test_data/inv_folding/9jl9.pdb"
  
  max_length: 512
```

#### Override Examples

```bash
# Generate multiple trials for better results
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_forward_folding \
    generation.n_trials=5

# Change number of diffusion steps
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_forward_folding \
    generation.nsteps=200
```

## Benchmark Results

### Unconditional Generation

Results from large-scale unconditional generation with self-reflection (100 samples per length):

| Model | Length | Total Structures | RMSD<2.0 | % Pass | Clusters | Diversity % | Avg TM | Avg RMSD | Avg pLDDT |
|-------|--------|-----------------|----------|--------|----------|-------------|--------|----------|-----------|
| genUME 90M | 100    | 100             | 85       | 85.0%  | 25       | 25.0%       | 0.8203 | 1.963    | 0.7111    |
| genUME 90M | 200    | 100             | 63       | 63.0%  | 19       | 19.0%       | 0.8043 | 2.467    | 0.6639    |
| genUME 90M | 300    | 100             | 62       | 62.0%  | 23       | 23.0%       | 0.8447 | 2.015    | 0.6851    |
| genUME 90M | 400    | 100             | 56       | 56.0%  | 13       | 13.0%       | 0.8505 | 2.191    | 0.7177    |
| genUME 90M | 500    | 91              | 31       | 34.1%  | 10       | 11.0%       | 0.8344 | 2.730    | 0.7283    |

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

### Inverse Folding

Performance on sequence design for given structures:

| Task | Model | AAR | TM-Score |
|------|-------|-----|----------|
| Inverse Folding | genUME 90M | 50.67% | 0.83 |

**Metrics Explanation:**
- **AAR (Amino Acid Recovery)**: Percentage of positions where the designed sequence matches the native sequence
- **TM-Score**: Structural similarity between input structure and structure predicted from designed sequence

**Key Observations:**
- AAR of 50.67% demonstrates strong sequence recovery capability
- TM-score of 0.83 indicates excellent structural preservation
- Model successfully designs sequences that fold back to target structures

**Dataset:** Benchmarked on the dataset from [Generative Flows on Discrete State-Spaces](https://arxiv.org/abs/2402.04997) (Campbell et al., ICML 2024)

### Forward Folding

Performance on structure prediction from sequences:

| Task | Model | TM-Score |
|------|-------|----------|
| Forward Folding | genUME 90M | 0.70 |

**Metrics Explanation:**
- **TM-Score**: Structural similarity between generated structure and reference structure

**Key Observations:**
- TM-score of 0.70 indicates good structure prediction capability
- Model generates plausible structures from sequence inputs

**Dataset:** Benchmarked on the dataset from [Generative Flows on Discrete State-Spaces](https://arxiv.org/abs/2402.04997) (Campbell et al., ICML 2024)

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
Begin with small test runs to validate your setup:
```bash
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name generate_unconditional \
    generation.length="[100]" \
    generation.num_samples=2 \
    generation.nsteps=100
```

### 2. Use Self-Reflection for Quality
Self-reflection is enabled by default in `generate_unconditional.yaml` to improve ESMFold metrics. To disable it:
```bash
generation.enable_self_reflection=false
```

### 3. ESMFold Validation
ESMFold is enabled by default in the provided configs and provides crucial quality metrics. Adjust `max_length` based on your sequences:
```bash
generation.max_length=1024  # For longer sequences
```

### 4. Batch Size Selection
- **GPU Memory Limited**: Use `batch_size: 1`
- **Long sequences (>400)**: Use `batch_size: 1`
- **Short sequences (<200)**: Can use `batch_size: 2-4`

### 5. Output Organization
Always use descriptive output directories for tracking experiments:
```bash
output_dir="./examples/generation_20251104_my_experiment"
```

### 6. Reproducibility
Set a seed for reproducible results:
```bash
seed=12345
```

### 7. Monitor Progress
CSV metrics and plots are enabled by default in the provided configs. To disable:
```bash
generation.save_csv_metrics=false
generation.create_plots=false
```

### 8. Multi-Chain Design
For inverse folding of multi-chain complexes:
```bash
generation.esmfold_chain_groups="[[A,B],[C]]"  # Design chains A+B together, C separately
```

### 9. Quality Control
Quality control is enabled by default in unconditional generation with self-reflection. Adjust thresholds if needed:
```bash
generation.self_reflection.quality_control.min_tm_score_forward=0.9
generation.self_reflection.quality_control.max_retries=50
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

### Benchmark Dataset

The inverse folding and forward folding benchmarks use the dataset from:

```bibtex
@inproceedings{campbell2024generative,
  title={Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design},
  author={Campbell, Andrew and Yim, Jason and Barzilay, Regina and Rainforth, Tom and Jaakkola, Tommi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024},
  url={https://arxiv.org/abs/2402.04997}
}
```

## Protein-Ligand

Gen-UME Protein-Ligand extends the model to handle protein-ligand complexes. The key insight is that **providing ligand context can improve both sequence design and structure prediction**, particularly for binding pocket residues.

### Protein-Ligand Inverse Folding

Design protein sequences conditioned on both protein structure **and** ligand context.

**Command Line:**

```bash
# Evaluate inverse folding with/without ligand context
uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
    --checkpoint /cv/scratch/u/lisanzas/gen_ume_protein_ligand/runs//2026-01-27T16-05-28/epoch=114-step=37186-val_loss=3.1698.ckpt \
    --data_dir /cv/data/ai4dd/data2/lisanzas/pdb_bind_12_15_25/test/ \
    --structure_path ./protein_ligand_eval_inverse_folding/ \
    --output protein_ligand_inverse_folding_results.csv \
    --pocket_threshold 5.0 \
    --num_samples 100 \
    --nsteps 100 \
    --device cuda \
    --decode_structure \
    --save_gt_structure \
    --minimize_ligand
```

**Options:**
- `--checkpoint`: Path to protein-ligand model checkpoint (required)
- `--data_dir`: Directory with `*_protein.pt` and `*_ligand.pt` pairs (default: `/data2/lisanzas/pdb_bind_12_15_25/test/`)
- `--structure_path`: Output directory for designed sequences (FASTA files), decoded structures (PDB files), and results
- `--output`: Output CSV file for per-structure results (default: `protein_ligand_inverse_folding_results.csv`)
- `--pocket_threshold`: Distance (Å) to define binding pocket residues (default: 5.0)
- `--num_samples`: Number of samples to evaluate (-1 for all, default: 100)
- `--nsteps`: Diffusion steps for generation (default: 100)
- `--device`: Device for computation, `cuda` or `cpu` (default: `cuda`)
- `--decode_structure`: Flag to decode and save predicted protein structures as PDB files (includes decoded ligand when ligand context is used)
- `--save_gt_structure`: Flag to save ground truth protein and protein-ligand complex structures as PDB files
- `--minimize_ligand`: Flag to apply Open Babel geometry correction to decoded ligand structures
- `--minimize_mode`: Minimization mode: `bonds_only`, `bonds_and_angles` (default), `local`, or `full`
- `--force_field`: Force field for minimization: `MMFF94` (default), `UFF`, `MMFF94s`, etc.
- `--minimize_steps`: Maximum number of minimization steps (default: 500)

**Output Files** (in `--structure_path`):
```
protein_ligand_eval/
├── protein_ligand_inverse_folding_results.csv  # Per-structure metrics
├── {pdb_id}_sequences.fasta                    # Ground truth + designed sequences
├── {pdb_id}_protein.pdb                        # Ground truth protein structure (--save_gt_structure)
├── {pdb_id}_complex.pdb                        # Ground truth protein-ligand complex (--save_gt_structure)
├── {pdb_id}_decoded_no_ligand.pdb              # Decoded structure without ligand context (--decode_structure)
├── {pdb_id}_decoded_with_ligand.pdb            # Decoded protein-ligand complex with ligand context (--decode_structure)
└── ...
```

**Tracked Metrics:**
- `aar_overall_*`: Overall amino acid recovery
- `aar_pocket_*`: Pocket-only recovery (residues within threshold of ligand)
- `aar_nonpocket_*`: Non-pocket recovery
- `*_delta`: Improvement from providing ligand context

### Protein-Ligand Forward Folding

Predict protein structure from sequence **with ligand context**.

**Command Line:**

```bash
# Evaluate forward folding with/without ligand context
uv run python -m lobster.cmdline.evaluate_protein_ligand_forward_folding \
    --checkpoint /cv/scratch/u/lisanzas/gen_ume_protein_ligand/runs//2026-01-27T16-05-28/epoch=114-step=37186-val_loss=3.1698.ckpt \
    --data_dir /cv/data/ai4dd/data2/lisanzas/pdb_bind_12_15_25/test/ \
    --structure_path ./protein_ligand_eval/ \
    --output protein_ligand_forward_folding_results.csv \
    --pocket_threshold 5.0 \
    --num_samples 100 \
    --nsteps 100 \
    --device cuda \
    --save_structures \
    --save_gt_structure \
    --minimize_ligand
```

**Options:**
- `--checkpoint`: Path to protein-ligand model checkpoint (required)
- `--data_dir`: Directory with `*_protein.pt` and `*_ligand.pt` pairs (default: `/data2/lisanzas/pdb_bind_12_15_25/test/`)
- `--structure_path`: Output directory for predicted structures (PDB files) and results
- `--output`: Output CSV file for per-structure results (default: `protein_ligand_forward_folding_results.csv`)
- `--pocket_threshold`: Distance (Å) to define binding pocket residues (default: 5.0)
- `--num_samples`: Number of samples to evaluate (-1 for all, default: 100)
- `--nsteps`: Diffusion steps for generation (default: 100)
- `--device`: Device for computation, `cuda` or `cpu` (default: `cuda`)
- `--temperature_seq`: Temperature for sequence sampling (default: 0.5)
- `--temperature_struc`: Temperature for structure sampling (default: 0.5)
- `--save_structures`: Flag to save predicted protein structures as PDB files
- `--save_gt_structure`: Flag to save ground truth protein and protein-ligand complex structures as PDB files
- `--minimize_ligand`: Flag to apply Open Babel geometry correction to decoded ligand structures
- `--minimize_mode`: Minimization mode: `bonds_only`, `bonds_and_angles` (default), `local`, or `full`
- `--force_field`: Force field for minimization: `MMFF94` (default), `UFF`, `MMFF94s`, etc.
- `--minimize_steps`: Maximum number of minimization steps (default: 500)

**Output Files** (in `--structure_path`):
```
protein_ligand_eval/
├── protein_ligand_forward_folding_results.csv  # Per-structure metrics
├── {pdb_id}_gt_protein.pdb                     # Ground truth protein structure (--save_gt_structure)
├── {pdb_id}_gt_complex.pdb                     # Ground truth protein-ligand complex (--save_gt_structure)
├── {pdb_id}_pred_no_ligand.pdb                 # Predicted structure without ligand context (--save_structures)
├── {pdb_id}_pred_with_ligand.pdb               # Predicted protein + decoded ligand structure (--save_structures)
└── ...
```

**Callback Configuration** (for training-time evaluation):

```yaml
protein_ligand_forward_folding:
  _target_: lobster.callbacks.ProteinLigandForwardFoldingCallback
  data_dir: /cv/data/ai4dd/data2/lisanzas/pdb_bind_12_15_25/test/
  structure_path: ${paths.output_dir}/protein_ligand_eval/
  save_every_n: 1000
  num_samples: 100
  pocket_distance_threshold: 5.0  # Å
  nsteps: 100
  # Ligand minimization options (for decoded ligand structures)
  minimize_ligand: false
  minimize_mode: "bonds_and_angles"  # "bonds_only", "bonds_and_angles", "local", or "full"
  force_field: "MMFF94"
  minimize_steps: 500
```

**Tracked Metrics:**
- `tm_score_*`: Overall TM-score (structure similarity)
- `rmsd_overall_*`: Overall backbone RMSD
- `rmsd_pocket_*`: Pocket-only RMSD (residues within threshold of ligand)
- `rmsd_nonpocket_*`: Non-pocket RMSD
- `*_delta`: Improvement from providing ligand context

### Protein-Ligand Checkpoints

**Latest checkpoint** (`/cv/scratch/u/lisanzas/gen_ume_protein_ligand/runs/2026-01-22T03-49-10/`):

| Checkpoint | Validation Loss |
|------------|-----------------|
| `epoch=49-step=16126-val_loss=3.5112.ckpt` | **3.5112** (best) |
| `epoch=57-step=18718-val_loss=3.5529.ckpt` | 3.5529 |
| `epoch=47-step=15303-val_loss=3.5601.ckpt` | 3.5601 |

### Data Format

The protein-ligand evaluators expect paired `.pt` files:

```
data_dir/
├── pdb_id_protein.pt  # Contains: coords_res, sequence, mask, indices
└── pdb_id_ligand.pt   # Contains: atom_coords, atom_names/element_indices, mask, bond_matrix
```

## Support

For issues and questions:
- **GitHub Issues**: [prescient-design/lobster](https://github.com/prescient-design/lobster)
- **Documentation**: See `src/lobster/cmdline/generate.py` for implementation details
- **Examples**: Check `src/lobster/hydra_config/experiment/` for example configurations

---

**Last Updated**: January 2026

