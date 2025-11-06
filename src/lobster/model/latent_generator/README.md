# LatentGenerator

A powerful protein and protein-ligand structure representation learning model for both continous and discrete representations.

## Table of Contents
- [Performance](#performance)
  - [Reconstruction Quality on CASP15 Proteins](#reconstruction-quality-on-casp15-proteins)
  - [Reconstruction Quality with Canonical Pose (Mol Frame)](#reconstruction-quality-with-canonical-pose-mol-frame)
  - [Fold Prediction Accuracy](#fold-prediction-accuracy)
- [Setup](#setup)
  - [Environment Setup](#environment-setup)
- [Getting Embeddings and Tokens](#getting-embeddings-and-tokens)
  - [Protein Example](#protein-example)
  - [Ligand Example](#ligand-example)
  - [Protein-Ligand Complex Example](#protein-ligand-complex-example)
  - [Command-line Example](#command-line-example)
- [Training](#training)
  - [Protein-only Training](#protein-only-training)
  - [Protein+Ligand (Complex) Training](#proteinligand-complex-training)
- [Model Configurations](#model-configurations)
  - [Ligand Models](#ligand-models)
  - [Protein-Ligand Models](#protein-ligand-models)
  - [Protein-Only Models](#protein-only-models)

## Performance

### Reconstruction Quality on CASP15 Proteins

We evaluated the reconstruction quality of our models on CASP15 proteins ≤ 512 residues. The table below shows the average RMSD between original and reconstructed structures:

**Evaluation Set**: CASP15 proteins ≤ 512 residues 

| Model | Tokens | Average RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) |
|-------|--------|------------------|--------------|--------------|--------------|
| LG full attention | 256 | 1.707 | 0.643 | 0.839 | 3.434 |

### Reconstruction Quality with Canonical Pose (Mol Frame)

We also evaluated the models using canonical pose mode, which makes the model invariant to rotations and translations:

**Evaluation Set**: CASP15 proteins ≤ 512 residues 

| Model | Tokens | Average RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) |
|-------|--------|------------------|--------------|--------------|--------------|
| LG full attention | 256 | 1.645 | 0.573 | 0.664 | 2.901 |


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

model_name = 'LG Ligand 20A'

# Load model with ligand support using the ModelInfo dataclass structure
load_model(
    methods[model_name].model_config.checkpoint, 
    methods[model_name].model_config.config_path, 
    methods[model_name].model_config.config_name, 
    overrides=methods[model_name].model_config.overrides
)

# Load protein-ligand complex
pdb_data = {"protein_coords": None, "protein_mask": None, "protein_seq": None} 
ligand_data = load_ligand("src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf")
pdb_data["ligand_coords"] = ligand_data["atom_coords"]
pdb_data["ligand_mask"] = ligand_data["mask"]
pdb_data["ligand_residue_index"] = ligand_data["atom_indices"]
pdb_data["ligand_atom_names"] = ligand_data["atom_names"]
pdb_data["ligand_indices"] = ligand_data["atom_indices"]
# Get tokens for the complex
tokens, embeddings = encode(pdb_data, return_embeddings=True)
print(tokens["ligand_tokens"].shape)  # (batch, length_ligand, n_tokens)
print(embeddings.shape) # (batch, length_protein+length_ligand, embedding_dim) 

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


### Protein-Ligand Complex Example (warning ligand recon not good yet)
```python
from lobster.model.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.io import writepdb_ligand_complex, load_pdb, load_ligand 
import torch

model_name = 'LG Ligand 20A seq 3di Aux'

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
python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG full attention' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/7kdr_protein.pdb \
    --decode

# Get tokens and decode to structure for ligand
python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Ligand 20A' \
    --ligand_path src/lobster/model/latent_generator/example/example_pdbs/4erk_ligand.sdf  \
    --decode
    
# Get tokens and decode to structure for protein-ligand
python src/lobster/model/latent_generator/cmdline/inference.py \
    --model_name 'LG Ligand 20A seq 3di Aux' \
    --pdb_path src/lobster/model/latent_generator/example/example_pdbs/4erk_protein.pdb \
    --ligand_path latent_generator/example/example_pdbs/4erk_ligand.sdf  \
    --decode

# Get embeddings (requires Python API)
```

The tokens are discrete representations that can be used for tasks like discrete generation (with LLMs or PLMs) and compact storage of structure information, while embeddings are continuous representations useful for tasks like similarity search, feature extraction, and representation centric tasks.

## Training

LatentGenerator training for protein-only models using the unified Lobster training framework.

### Quick Start

Train a latent_generator model using the provided SLURM script:

```bash
# Submit training job
sbatch slurm/scripts/train_latent_generator.sh
```

Or run locally without SLURM:

```bash
# Set required environment variables
export LOBSTER_RUNS_DIR="/path/to/runs"
export LOBSTER_DATA_DIR="/path/to/data"

# Train with default experiment configuration
uv run lobster_train experiment=train_latent_generator

# Override specific parameters
uv run lobster_train experiment=train_latent_generator \
    data.batch_size=32 \
    model.quantizer.n_tokens=512 \
    trainer.devices=4
```

### Configuration Files

The training uses the unified Lobster configuration system:

- **Experiment config**: `src/lobster/hydra_config/experiment/train_latent_generator.yaml`
- **Model config**: `src/lobster/hydra_config/model/latent_generator.yaml`
- **Data config**: `src/lobster/hydra_config/data/structure_pdb.yaml`

### Key Configuration Parameters

Based on the actual model configuration (`src/lobster/hydra_config/model/latent_generator.yaml`):

**Model Architecture:**
- `quantizer.n_tokens`: Number of discrete tokens in codebook (default: 256)
- `structure_encoder.embed_dim_hidden`: Hidden embedding dimension (default: 256)
- `structure_encoder.uvit_n_layers`: Number of encoder layers (default: 6)
- `structure_encoder.uvit_n_heads`: Number of attention heads (default: 8)
- `structure_encoder.data_fixed_size`: Maximum sequence length (default: 512)
- `structure_encoder.n_atoms`: Number of atoms per residue (default: 3 for backbone)

**Training Parameters:**
- `num_warmup_steps`: Learning rate warmup steps (default: 5000)
- `num_training_steps`: Total training steps (default: 50000)
- `optim.lr`: Learning rate (default: 1e-4)

**Decoder Configuration:**
- `decoder_factory.decoder_mapping.vit_decoder.struc_token_dim`: Decoder token dimension (default: 512)
- `decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size`: Token codebook size (default: 256)

**Loss Configuration:**
- `loss_factory.weight_dict.pairwise_l2_loss`: Weight for pairwise L2 loss (default: 1.0)
- `loss_factory.weight_dict.l2_loss`: Weight for L2 loss (default: 0.01)

### SLURM Training Configuration

The SLURM script (`slurm/scripts/train_latent_generator.sh`) is configured for multi-GPU training:

**Hardware Configuration:**
- 1 node with 8 GPUs (b200 partition)
- 16 CPUs per GPU task
- 256GB RAM
- 7-day maximum runtime

**Environment Variables:**
- `LOBSTER_RUNS_DIR`: Directory for saving checkpoints and logs
- `LOBSTER_DATA_DIR`: Directory for caching data
- `LOBSTER_USER`: WandB username for logging
- `WANDB_BASE_URL`: WandB server URL (if using custom instance)

**Training Settings:**
- DDP strategy with `ddp_find_unused_parameters_true`
- 8 data workers per GPU
- BF16 mixed precision
- Gradient accumulation: 16 batches

Edit the SLURM script to adjust these settings for your cluster.

### Dataset Preparation

The latent_generator expects protein structure datasets with:
- Backbone atom coordinates (N, CA, C atoms)
- Residue information  
- Sequence data

Configure your dataset paths in `src/lobster/hydra_config/data/structure_pdb.yaml`:

```yaml
path_to_datasets: [
  "/path/to/train.pt",
  "/path/to/validation.pt", 
  "/path/to/test.pt"
]
batch_size: 40
num_workers: 12
```

Or override on the command line:

```bash
uv run lobster_train experiment=train_latent_generator \
    data.path_to_datasets="['/path/to/train.pt','/path/to/val.pt','/path/to/test.pt']"
```

### Resuming from Checkpoint

Load a checkpoint to resume training or fine-tune:

```bash
# Resume training from checkpoint
uv run lobster_train experiment=train_latent_generator \
    model.ckpt_path=/path/to/checkpoint.ckpt

# Or via SLURM (modify the script to add):
# model.ckpt_path=/path/to/checkpoint.ckpt
```

### Monitoring Training

Training progress can be monitored through:

1. **SLURM Output**: Job logs are saved to the path specified in the SLURM script (default: `/data2/ume/latent_generator_/slurm/logs/train/`)
2. **WandB**: Logs automatically to the `lobster_latent_generator` project
   - Run name includes: token count, encoder/decoder dimensions, dataset name, and timestamp
   - Example: `latent_generator-tokens_256-enc_256-dec_512_pdb_2024-01-01T12-00-00`
3. **Console Output**: Real-time loss and metrics (when running locally)

## Model Configurations

LatentGenerator provides several pre-configured models optimized for different use cases. These configurations include all necessary settings and overrides, making them easy to use without manual configuration.

### Ligand Models

#### LG Ligand 20A
- **Description**: Ligand only model with 20Å spatial attention
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Ligand only decoder
  - 512 ligand tokens
- **Use Case**: Ligand analysis and generation

#### LG Ligand 20A 512 1024
- **Description**: Ligand only model with 20Å spatial attention
- **Features**:
  - 512-dim embeddings
  - 20Å spatial attention
  - Ligand only decoder
  - 1024 ligand tokens
- **Use Case**: High-dimensional ligand analysis and generation

#### LG Ligand 20A 512 1024 element
- **Description**: Ligand only model with 20Å spatial attention and element awareness
- **Features**:
  - 512-dim embeddings
  - 20Å spatial attention
  - Ligand only decoder with element awareness
  - 1024 ligand tokens
- **Use Case**: Element-aware ligand analysis and generation

#### LG Ligand 20A continuous
- **Description**: Ligand only model with 20Å spatial attention and continuous encoding
- **Features**:
  - 512-dim embeddings
  - 20Å spatial attention
  - Ligand only decoder
  - Continuous ligand encoding (no quantization)
- **Use Case**: Continuous ligand representation learning

### Protein-Ligand Models

#### LG Ligand 20A seq 3di Aux
- **Description**: Protein-ligand model with sequence and 3Di awareness
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence and 3Di decoder
  - Ligand encoding support
  - 512 ligand tokens
  - 512 protein tokens
- **Use Case**: Protein-ligand complex analysis and generation with sequence awareness

### Protein-Only Models


#### LG full attention
- **Description**: Full attention model without spatial masking
- **Features**:
  - Standard configuration
  - Full attention (no spatial masking)
  - 256 protein tokens
- **Use Case**: Global protein structure analysis

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
python latent_generator/cmdline/inference.py --model_name 'LG full attention' --pdb_path your_protein.pdb

# Using custom checkpoint
python latent_generator/cmdline/inference.py \
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
