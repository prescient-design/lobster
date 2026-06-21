# LatentGenerator

A powerful protein and protein-ligand structure representation learning model for both continous and discrete representations.

## Table of Contents
- [Performance](#performance)
  - [Structure Reconstruction Quality on CASP15 Proteins](#structure-reconstruction-quality-on-casp15-proteins)
  - [Ligand Reconstruction Quality](#ligand-reconstruction-quality)
  - [Protein-Ligand Complex Reconstruction Quality](#protein-ligand-complex-reconstruction-quality)
- [Variants](#variants)
- [Setup](#setup)
  - [Environment Setup](#environment-setup)
- [Getting Embeddings and Tokens](#getting-embeddings-and-tokens)
  - [Protein Example](#protein-example)
  - [Ligand Example](#ligand-example)
  - [Protein-Ligand Complex Example](#protein-ligand-complex-example)
  - [Command-line Example](#command-line-example)
  - [Ligand Structure Minimization](#ligand-structure-minimization)
- [Evaluation](#evaluation)
  - [Evaluating Reconstruction Quality on CASP15](#evaluating-reconstruction-quality-on-casp15)
- [Model Configurations](#model-configurations)
  - [Protein-Ligand Models](#protein-ligand-models)
  - [Protein-Only Models](#protein-only-models)

## Performance

### Structure Reconstruction Quality on CASP15 Proteins

We evaluated the reconstruction quality of our models on CASP15 proteins (≤ 512 residues). The continuous baseline establishes an upper bound for the ViT architecture.

**Evaluation Set**: CASP15 proteins ≤ 512 residues

| Model | Quantizer | Size | RMSD (Å) | Std | Min | Max |
|-------|-----------|------|----------|-----|-----|-----|
| LG Protein (cont.) | None | - | 0.462 | 0.322 | 0.200 | 1.271 |
| LG Protein SLQ | SLQ | 256 | 1.647 | 0.535 | 0.979 | 3.189 |
| LG Prot-Lig SLQ | SLQ | 256 | 1.873 | 1.054 | 0.798 | 5.143 |
| LG Prot-Lig SLQ | SLQ | 4096 | 3.097 | 2.009 | 1.242 | 8.474 |
| LG Protein FSQ | FSQ | 240 | 1.848 | 1.194 | 0.483 | 5.419 |
| LG Prot-Lig FSQ | FSQ | 4375 | 1.260 | 0.632 | 0.651 | 3.117 |
| LG Prot-Lig FSQ | FSQ | 4375/15360 | 1.418 | 0.810 | 0.748 | 3.396 |

### Ligand Reconstruction Quality

We evaluated ligand reconstruction quality on 30,936 ligand structures from the GEOM dataset. The unified protein-ligand model achieves comparable performance to the specialist ligand-only model, demonstrating the architecture's capacity to handle multimodal distributions within a shared parameter set.

**Evaluation Set**: 30,936 ligands from GEOM dataset

| Model | Size | Avg RMSD (Å) | Std | Min | Max |
|-------|------|--------------|-----|-----|-----|
| LG Ligand SLQ | 512 | 0.752 | 0.305 | 0.065 | 4.943 |
| LG Prot-Lig SLQ | 512 | 0.920 | 0.236 | 0.152 | 3.704 |
| LG Prot-Lig SLQ | 4096 | 1.239 | 0.335 | 0.196 | 4.101 |
| LG Prot-Lig FSQ | 4375 | 0.395 | 0.059 | 0.179 | 1.784 |
| LG Prot-Lig FSQ | 15360 | 0.295 | 0.052 | 0.120 | 1.792 |

### Protein-Ligand Complex Reconstruction Quality

Comparison of FSQ and SLQ variants on PDBbind complexes. Token counts represent the specific codebook size for protein and ligand components respectively.

**Evaluation Set**: PDBbind complexes

| Model | Metric | Prot Tokens | Lig Tokens | Alignment | Avg RMSD (Å) | Std | Min | Max |
|-------|--------|-------------|------------|-----------|--------------|-----|-----|-----|
| LG Prot-Lig SLQ | Ligand | 256 | 512 | Individual | 1.411 | 0.593 | 0.365 | 4.519 |
| LG Prot-Lig SLQ | Ligand | 4096 | 4096 | Individual | 1.620 | 0.711 | 0.533 | 6.756 |
| LG Prot-Lig FSQ | Ligand | 4375 | 4375 | Individual | 0.705 | 0.139 | 0.345 | 1.935 |
| LG Prot-Lig FSQ | Ligand | 4375 | 15360 | Individual | 0.657 | 0.146 | 0.315 | 2.407 |
| | | | | | | | | |
| LG Prot-Lig SLQ | Complex | 256 | 512 | Joint | 1.567 | 0.343 | 0.939 | 5.579 |
| LG Prot-Lig SLQ | Complex | 4096 | 4096 | Joint | 4.680 | 2.962 | 1.415 | 19.173 |
| LG Prot-Lig FSQ | Complex | 4375 | 4375 | Joint | 1.011 | 0.127 | 0.723 | 2.387 |
| LG Prot-Lig FSQ | Complex | 4375 | 15360 | Joint | 1.009 | 0.138 | 0.739 | 3.578 |
| | | | | | | | | |
| LG Prot-Lig SLQ | Ligand | 256 | 512 | Joint (c) | 2.306 | 0.758 | 0.711 | 5.927 |
| LG Prot-Lig SLQ | Ligand | 4096 | 4096 | Joint (c) | 3.589 | - | - | - |
| LG Prot-Lig FSQ | Ligand | 4375 | 4375 | Joint (c) | 1.011 | 0.271 | 0.507 | 3.729 |
| LG Prot-Lig FSQ | Ligand | 4375 | 15360 | Joint (c) | 0.998 | - | - | - |

## Variants

Two Lightning modules live under
[`lobster.model.latent_generator.tokenizer`](tokenizer/). They share the
same Hydra-wired `DecoderFactory` / `LossFactory` building blocks and
checkpoint format but differ in what the input alphabet is:

| Variant | Input | Pipeline | Quantizer | Use case |
|---|---|---|---|---|
| [`TokenizerMulti`](tokenizer/_tokenizer_multi.py) | `coords_res` `(B, L, 3, 3)` | encoder → quantizer → decoder | yes (SLQ / FSQ / …) | Canonical LG: backbone in, discrete token bottleneck, backbone out. Powers all the checkpoints in the "Model Configurations" tables below. |
| [`Tokenizer3diInputFlow`](tokenizer/_tokenizer_3di_input_flow.py) | `3di_states` `(B, L)` long (Foldseek 20-state alphabet, recomputed at data-load time by [`Structure3diTransform`](../../transforms/_structure_transforms.py)) | flow-matching SDE sampler over a learned U-ViT velocity field, conditioned on the 3Di tokens | no (3Di tokens themselves are the discrete bottleneck) | Generative reconstruction of backbone coords from a model-agnostic 3Di string. Trained as continuous flow matching with classifier-free guidance and an auxiliary differentiable mini3di CE-from-coords loss. Inference uses a 200-step power-2 SDE schedule at guidance scale `w=2.0` (the published "WINNER_W2" recipe). Published checkpoint: [`LG 3Di Flow Decoder`](#lg-3di-flow-decoder) below. |

The `Tokenizer3diInputFlow` training surface ships with one model yaml
family + matching experiment yamls under
[`hydra_config/model/`](../../hydra_config/model/) and
[`hydra_config/experiment/`](../../hydra_config/experiment/) — see the
files matching `latent_generator_3di_input_flow_nokabsch_velocity_base*.yaml`
and the corresponding `train_*` experiments. The data side reuses
[`data/structure_pdb_with_3di.yaml`](../../hydra_config/data/structure_pdb_with_3di.yaml).

This variant is **not** a substitutable codec for LeFlur today —
LeFlur expects an LG codec that exposes `encode_structure` returning
discrete tokens, whereas `Tokenizer3diInputFlow` returns coordinates
directly via flow-matching sampling. A future LeFlur integration would
need a separate path.

## Setup

### Environment Setup
On
```bash
# With latent generator CPU support
uv sync --extra struct-cpu

# With latent generator GPU support  
uv sync --extra struct-gpu
```

## Getting Embeddings and Tokens

You can extract both embeddings and tokens from a trained LatentGenerator model using either Python or the command line.

### Protein Example
```python
from lobster.model.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.io import writepdb, writepdb_ligand_complex, load_pdb
import torch


model_name = 'LG full attention'

# Load model using the ModelInfo dataclass structure
load_model(
    methods[model_name].model_config.checkpoint, 
    methods[model_name].model_config.config_path, 
    methods[model_name].model_config.config_name, 
    overrides=methods[model_name].model_config.overrides
)

# Load a PDB file
pdb_data = load_pdb("src/lobster/model//latent_generator/example/example_pdbs/7kdr_protein.pdb")

# Get tokens (discrete representations) and embeddings (continuous representations)
tokens, embeddings = encode(pdb_data, return_embeddings=True)
print(tokens.shape)  # (batch, length, n_tokens)
print(embeddings.shape)  # (batch, length, embedding_dim)

# Decode tokens back to structure
decoded_outputs = decode(tokens, x_emb=embeddings)
seq = torch.zeros(decoded_outputs[0].shape[1], dtype=torch.long)[None]
writepdb("decoded.pdb", decoded_outputs[0], seq[0])

```

### Ligand Example
```python
from lobster.model.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.io import writepdb_ligand_complex, load_pdb, load_ligand 
import torch

model_name = 'LG Protein Ligand fsq 4375'

# Load model with ligand support using the ModelInfo dataclass structure
load_model(
    methods[model_name].model_config.checkpoint, 
    methods[model_name].model_config.config_path, 
    methods[model_name].model_config.config_name, 
    overrides=methods[model_name].model_config.overrides
)

# Load ligand only (no protein)
pdb_data = {"protein_coords": None, "protein_mask": None, "protein_seq": None} 
ligand_data = load_ligand("src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf")
pdb_data["ligand_coords"] = ligand_data["atom_coords"]
pdb_data["ligand_mask"] = ligand_data["mask"]
pdb_data["ligand_residue_index"] = ligand_data["atom_indices"]
pdb_data["ligand_atom_names"] = ligand_data["atom_names"]
pdb_data["ligand_indices"] = ligand_data["atom_indices"]

# Get tokens for the ligand
tokens, embeddings = encode(pdb_data, return_embeddings=True)
print(tokens["ligand_tokens"].shape)  # (batch, length_ligand, n_tokens)
print(embeddings.shape) # (batch, length_ligand, embedding_dim) 

# Decode tokens back to structure
decoded_outputs = decode(tokens, x_emb=embeddings)

# Save the reconstructed ligand
writepdb_ligand_complex(
  "decoded_ligand.pdb", 
  ligand_atoms=decoded_outputs[0]["ligand_coords"][0],
  ligand_atom_names=None,  # Optional: provide atom names if available
  ligand_chain="L",
  ligand_resname="LIG")

```

### Protein-Ligand Complex Example

```python
from lobster.model.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.io import writepdb_ligand_complex, load_pdb, load_ligand 
import torch

# Choose one of the protein-ligand models:
# - 'LG Protein Ligand fsq 4375' (4375 tokens for both protein and ligand)
# - 'LG Protein Ligand fsq 4375 15360' (4375 protein tokens, 15360 ligand tokens)
model_name = 'LG Protein Ligand fsq 4375'

# Load model with ligand support using the ModelInfo dataclass structure
load_model(
    methods[model_name].model_config.checkpoint, 
    methods[model_name].model_config.config_path, 
    methods[model_name].model_config.config_name, 
    overrides=methods[model_name].model_config.overrides
)

# Load protein-ligand complex
pdb_data = load_pdb("src/lobster/model/latent_generator/example/example_pdbs/4erk_protein.pdb")  
ligand_data = load_ligand("src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf")
pdb_data["ligand_coords"] = ligand_data["atom_coords"]
pdb_data["ligand_mask"] = ligand_data["mask"]
pdb_data["ligand_residue_index"] = ligand_data["atom_indices"]
pdb_data["ligand_atom_names"] = ligand_data["atom_names"]
pdb_data["ligand_indices"] = ligand_data["atom_indices"]

# Get tokens for the complex
tokens, embeddings = encode(pdb_data, return_embeddings=True)
print(tokens["protein_tokens"].shape)  # (batch, length_protein, n_tokens)
print(tokens["ligand_tokens"].shape)  # (batch, length_ligand, n_tokens)
print(embeddings.shape) # (batch, length_protein+length_ligand, embedding_dim) 

# Decode tokens back to structure
decoded_outputs = decode(tokens, x_emb=embeddings)
decoded_outputs = decoded_outputs[0]
seq = torch.zeros(decoded_outputs['protein_coords'].shape[1], dtype=torch.long)[None]

# Save the reconstructed complex
writepdb_ligand_complex(
    "decoded_complex.pdb",
    ligand_atoms=decoded_outputs["ligand_coords"][0],
    ligand_atom_names=None,  # Optional: provide atom names if available
    ligand_chain="L",
    ligand_resname="LIG",
    protein_atoms=decoded_outputs["protein_coords"][0],
    protein_seq=seq[0]
)
```


### Command-line Example
```bash
# Get tokens and decode to structure for protein only
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG full attention' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/7kdr_protein.pdb \
    --decode

# Get tokens and decode to structure for ligand only
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Protein Ligand fsq 4375' \
    --ligand_path src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf \
    --decode

# Get tokens and decode to structure for protein-ligand complex using LG Protein Ligand fsq 4375
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Protein Ligand fsq 4375' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/4erk_protein.pdb \
    --ligand_path src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf \
    --decode

# Get tokens and decode using LG Protein Ligand fsq 4375 15360 (higher ligand resolution)
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Protein Ligand fsq 4375 15360' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/4erk_protein.pdb \
    --ligand_path src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf \
    --decode

# Get embeddings (requires Python API)
```

### Ligand Structure Minimization

For protein-ligand complexes, you can apply post-decoding geometry correction to improve ligand bond lengths and angles using Open Babel force fields. This is especially useful for improving the quality of decoded ligand structures.

```bash
# Decode with ligand minimization (bonds and angles correction - recommended)
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Protein Ligand fsq 4375' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/4erk_protein.pdb \
    --ligand_path src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf \
    --output_pdb decoded_complex.pdb \
    --decode \
    --minimize

# Specify output paths explicitly
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Protein Ligand fsq 4375' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/4erk_protein.pdb \
    --ligand_path src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf \
    --output_file_encode encoded_latents.pt \
    --output_file_decode decoded_outputs.pt \
    --output_pdb decoded_complex.pdb \
    --decode \
    --minimize
```

#### Minimization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--minimize` | False | Enable ligand structure minimization after decoding |
| `--minimize_mode` | `bonds_and_angles` | Minimization strategy (see below) |
| `--force_field` | `MMFF94` | Force field: `MMFF94`, `MMFF94s`, `UFF`, `GAFF`, `Ghemical` |
| `--minimize_steps` | `500` | Maximum optimization steps |
| `--minimize_method` | `cg` | Optimization method: `cg` (conjugate gradients) or `sd` (steepest descent) |

#### Minimization Modes

| Mode | Description |
|------|-------------|
| `bonds_and_angles` | **Recommended.** Constrained force field minimization that idealizes both bond lengths and angles while preserving overall structure. |
| `bonds_only` | Only corrects bond lengths to ideal values, preserving torsion angles. |

#### Example with Custom Minimization Settings

```bash
# Use UFF force field with bonds_only mode
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Protein Ligand fsq 4375' \
    --pdb_path protein.pdb \
    --ligand_path ligand.sdf \
    --output_pdb output.pdb \
    --decode \
    --minimize \
    --minimize_mode bonds_only \
    --force_field UFF
```

#### CONECT Records

When the ligand SDF file contains bond information, the output PDB will include CONECT records for proper bond visualization in molecular viewers like PyMOL, Chimera, or VMD.

The tokens are discrete representations that can be used for tasks like discrete generation (with LLMs or PLMs) and compact storage of structure information, while embeddings are continuous representations useful for tasks like similarity search, feature extraction, and representation centric tasks.

## Evaluation

### Evaluating Reconstruction Quality on CASP15

The `evaluate_reconstruction.py` script evaluates the reconstruction quality of LatentGenerator models by computing the aligned RMSD between original and reconstructed structures.

#### Basic Usage

Evaluate a single model on a directory of structures:

```bash
uv run python src/lobster/metrics/evaluate_reconstruction.py \
    --models "LG full attention" \
    --data_dir /path/to/casp15/structures/ \
    --output_file reconstruction_results.json
```

#### Using Canonical Pose (Mol Frame)

Evaluate with canonical pose mode for rotation/translation invariance:

```bash
uv run python src/lobster/metrics/evaluate_reconstruction.py \
    --models "LG full attention" \
    --data_dir /path/to/casp15/structures/ \
    --output_file reconstruction_canonical.json \
    --use_canonical_pose
```

#### Input File Formats

The evaluation script supports multiple structure file formats:
- **PDB files** (`.pdb`): Standard protein structure files
- **SDF files** (`.sdf`): Ligand structure files
- **PyTorch files** (`.pt`): Pre-processed structure data

#### Performance Metrics

The evaluation reports:
- **Average RMSD**: Mean reconstruction error across all structures
- **Std RMSD**: Standard deviation of RMSD values
- **Min/Max RMSD**: Best and worst reconstruction quality
- **Success Rate**: Number of successful vs. failed reconstructions

## Model Configurations

LatentGenerator provides pre-configured models optimized for different use cases. These configurations include all necessary settings and overrides, making them easy to use without manual configuration.

### Protein-Ligand Models

#### LG Protein Ligand fsq 4375
- **Description**: Protein-ligand model with FSQ quantization (4375 tokens)
- **Features**:
  - 5-dim embeddings
  - FSQ quantization
  - Ligand encoding support
  - 4375 ligand tokens
  - 4375 protein tokens
- **Use Case**: Protein-ligand complex analysis and generation with balanced token resolution

#### LG Protein Ligand fsq 4375 15360
- **Description**: Protein-ligand model with FSQ quantization (4375 protein tokens, 15360 ligand tokens)
- **Features**:
  - 5-dim embeddings
  - FSQ quantization
  - Ligand encoding support
  - 15360 ligand tokens (higher resolution for ligands)
  - 4375 protein tokens
- **Use Case**: Protein-ligand complex analysis and generation with higher ligand resolution

### Protein-Only Models

#### LG full attention
- **Description**: Full attention model without spatial masking
- **Features**:
  - Standard configuration
  - Full attention (no spatial masking)
  - 256 protein tokens
- **Use Case**: Global protein structure analysis

#### LG 3Di Flow Decoder
- **Description**: 3Di-conditioned flow-matching backbone decoder. Reconstructs N/Cα/C coordinates from per-residue Foldseek 3Di tokens via a 200-step power-2 SDE trajectory with classifier-free guidance (`w=2.0`).
- **Features**:
  - U-ViT 768d / 12L / 12h (~110M params)
  - Velocity-prediction continuous flow matching
  - Foldseek 3Di alphabet input (20 states)
  - No encoder, no quantizer — pure decoder
  - Backed by [`Tokenizer3diInputFlow`](tokenizer/_tokenizer_3di_input_flow.py) (separate class from `TokenizerMulti`)
- **Checkpoint**: [`Sidney-Lisanza/latent_generator/checkpoints_for_lg/LG_3Di_Flow_Decoder.ckpt`](https://huggingface.co/Sidney-Lisanza/latent_generator/blob/main/checkpoints_for_lg/LG_3Di_Flow_Decoder.ckpt) (~1.3 GB; metadata sidecar at `LG_3Di_Flow_Decoder.json`)
- **Reference metrics** (see Appendix "3Di-flow decoder reconstruction" of [LeFlur, Lisanza et al. 2026](https://openreview.net/forum?id=z5EwGneX36)):
  - 30-protein PDB val (T=200, w=2.0): Kab = 6.55 Å, 3Di Recovery = 76.5%
  - CASP15 (n=20, L≤512), single-shot: Kab = 9.40 Å, 3DR = 73.5%, TM = 0.644
  - CASP15 (n=20), best-of-10 (AAR-pick): Kab = 7.91 Å, 3DR = 78.4%, TM = 0.701
  - ProstT5 test (n=390, L≤512), single-shot: Kab = 11.99 Å, 3DR = 83.0%, TM = 0.531
- **Use Case**: Reconstruct backbone structure from a model-agnostic 3Di string (e.g. ProstT5 outputs, Foldseek searches). Note that 3Di tokens specify *local* geometry only; many distinct global folds map to the same 3Di string, so reconstruction quality is fundamentally bounded by the alphabet — see paper for the trade-off.

##### Quick-start CLI

The model is wired into the existing `inference.py` driver, so the
invocation matches every other LG variant:

```bash
# Reconstruct a single PDB through the encode → decode round-trip.
# The first call downloads ~1.3 GB into ~/.cache/lobster.
uv run python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG 3Di Flow Decoder' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/efhand.pdb \
    --output_pdb decoded.pdb \
    --decode
```

Internally: `Structure3diTransform` derives the 3Di tokens from the
input PDB, the published U-ViT decoder integrates a 200-step SDE
trajectory at `w=2.0`, and the resulting backbone is written to
`--output_pdb`. Sampling takes ~5 s / protein on an A10G at L≈100.

##### Python API

```python
from lobster.model.latent_generator.cmdline import (
    decode, encode, load_model, methods,
)
from lobster.model.latent_generator.io import load_pdb, writepdb

mc = methods["LG 3Di Flow Decoder"].model_config
load_model(
    mc.checkpoint, mc.config_path, mc.config_name,
    overrides=mc.overrides, model_class=mc.model_class,
)

pdb_data = load_pdb("src/lobster/model/latent_generator/example/example_pdbs/efhand.pdb")
# encode is a no-op for the 3Di flow decoder -- it returns pdb_data
# (and an empty embeddings tensor) to match the LG-codec API shape.
latents, _embeddings = encode(pdb_data, return_embeddings=True)
# decode runs Structure3diTransform internally and integrates the
# 200-step SDE trajectory. Returns (coords (B, L, 3, 3), None).
coords, _seq = decode(latents)
writepdb("decoded.pdb", coords[0], pdb_data["sequence"][0])
```

## Loading Models

### Using Pre-configured Models

To use any of these models, simply specify the model name when loading. The `methods` dictionary contains all pre-configured models with their checkpoints, configs, and overrides:

```python
from lobster.model.latent_generator.latent_generator.cmdline import load_model, methods

# Load a pre-configured model using the ModelInfo dataclass structure
model_name = 'LG full attention'
load_model(
    methods[model_name].model_config.checkpoint,
    methods[model_name].model_config.config_path,
    methods[model_name].model_config.config_name,
    overrides=methods[model_name].model_config.overrides
)
```

### Using Custom Checkpoints

You can also load custom checkpoints by providing the checkpoint path and configuration details directly:

```python
# Load a custom model
load_model(
    checkpoint_path="path/to/your/checkpoint.ckpt",
    cfg_path="path/to/config/",
    cfg_name="config_name",
    overrides=["+tokenizer.structure_encoder.embed_dim=256"]
)
```

### Command Line Usage

Or via command line:
```bash
# Using pre-configured model
uv run python latent_generator/cmdline/inference.py --model_name 'LG full attention' --pdb_path your_protein.pdb

# Using custom checkpoint
uv run python latent_generator/cmdline/inference.py \
    --ckpt_path path/to/your/checkpoint.ckpt \
    --cfg_path path/to/config/ \
    --cfg_name config_name \
    --pdb_path your_protein.pdb
```

### Supported Checkpoint Sources

The inference system supports multiple checkpoint sources:

- **Local files**: Direct path to checkpoint file
- **S3 URLs**: `s3://bucket-name/path/to/checkpoint.ckpt`
- **Hugging Face**: `https://huggingface.co/user/repo/resolve/main/checkpoint.ckpt`

The system will automatically download and cache checkpoints from remote sources.

## Citation

If you use LatentGenerator (or LeFlur, which builds on it) in your work, please cite:

> Sidney L. Lisanza, Karina Zadorozhny, Frederic A. Dreyer, and Kyunghyun Cho. **LeFlur: A Biomolecular Design Model with Latent Structure Tokens.** *The 2026 Workshop on Generative and Agentic AI for Biology*, 2026. <https://openreview.net/forum?id=z5EwGneX36>

```bibtex
@inproceedings{lisanza2026leflur,
  title     = {{LeFlur}: A Biomolecular Design Model with Latent Structure Tokens},
  author    = {Lisanza, Sidney L. and Zadorozhny, Karina and Dreyer, Frederic A. and Cho, Kyunghyun},
  booktitle = {The 2026 Workshop on Generative and Agentic AI for Biology},
  year      = {2026},
  url       = {https://openreview.net/forum?id=z5EwGneX36}
}
```
