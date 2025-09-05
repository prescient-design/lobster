# LatentGenerator

A powerful protein and protein-ligand structure representation learning model for both continous and discrete representations.

## Table of Contents
- [Performance](#performance)
  - [Reconstruction Quality on CASP15 Proteins](#reconstruction-quality-on-casp15-proteins)
  - [Reconstruction Quality with Canonical Pose (Mol Frame)](#reconstruction-quality-with-canonical-pose-mol-frame)
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

| Model | Average RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) |
|-------|------------------|--------------|--------------|--------------|
| LG full attention | 1.707 | 0.643 | 0.839 | 3.434 |
| LG 10A | 3.698 | 1.756 | 1.952 | 7.664 |
| LG 20A c6d Aux | 4.395 | 2.671 | 1.678 | 11.306 |
| LG 20A seq 3di c6d Aux | 4.428 | 1.723 | 2.757 | 8.556 |
| LG 20A 3di c6d Aux | 4.484 | 2.458 | 2.390 | 11.696 |
| LG 20A | 4.470 | 3.540 | 1.630 | 12.864 |
| LG 20A seq 3di c6d 512 Aux | 5.761 | 4.349 | 1.188 | 17.442 |
| LG 20A seq Aux | 5.449 | 2.862 | 3.063 | 13.342 |
| LG 20A seq 3di Aux | 6.112 | 3.723 | 2.973 | 17.839 |
| LG 20A 3di Aux | 7.844 | 4.289 | 3.119 | 16.500 |

### Reconstruction Quality with Canonical Pose (Mol Frame)

We also evaluated the models using canonical pose mode, which makes the model invariant to rotations and translations:

**Evaluation Set**: CASP15 proteins ≤ 512 residues 

| Model | Average RMSD (Å) | Std RMSD (Å) | Min RMSD (Å) | Max RMSD (Å) |
|-------|------------------|--------------|--------------|--------------|
| LG full attention | 1.645 | 0.573 | 0.664 | 2.901 |
| LG 10A | 4.005 | 2.173 | 1.981 | 9.883 |
| LG 20A c6d Aux | 4.603 | 3.028 | 1.240 | 12.297 |
| LG 20A seq 3di c6d Aux | 4.614 | 2.103 | 2.811 | 9.061 |
| LG 20A 3di c6d Aux | 4.140 | 2.108 | 2.195 | 9.275 |
| LG 20A | 4.268 | 3.306 | 1.461 | 12.989 |
| LG 20A seq 3di c6d 512 Aux | 5.445 | 3.963 | 1.568 | 15.305 |
| LG 20A seq Aux | 5.759 | 3.248 | 2.246 | 16.543 |
| LG 20A seq 3di Aux | 6.107 | 2.974 | 3.097 | 13.456 |
| LG 20A 3di Aux | 8.288 | 4.434 | 3.043 | 16.252 |

**Key Findings:**
- **LG full attention** achieves the best reconstruction quality in both modes (1.707 ± 0.643 Å standard, 1.645 ± 0.573 Å canonical)
- **LG 10A** performs well in both modes (3.698 ± 1.756 Å standard, 4.005 ± 2.173 Å canonical)
- **LG 20A 3di c6d Aux** shows improved performance with canonical pose (4.140 ± 2.108 Å vs 4.484 ± 2.458 Å)
- Canonical pose mode generally maintains or improves performance for most models

## Table of Contents
- [Setup](#setup)
- [Getting Embeddings and Tokens](#getting-embeddings-and-tokens)
- [Training](#training)
- [Model Configurations](#model-configurations)
- [Inference](#inference)

## Setup

### Environment Setup
```bash
# Create and activate virtual environment
uv venv latentgenerator
source latentgenerator/bin/activate

# Install dependencies
uv pip install torch Cython numpy
uv pip install -e . --no-build-isolation
```

## Getting Embeddings and Tokens

You can extract both embeddings and tokens from a trained LatentGenerator model using either Python or the command line.

### Protein Example
```python
from lobster.model.latent_generator.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.latent_generator.io import writepdb, writepdb_ligand_complex, load_pdb
import torch


model_name = 'LG 20A seq 3di c6d Aux'

# Load model (optionally with overrides)
load_model(methods[model_name]["model_config"]["checkpoint"], methods[model_name]["model_config"]["config_path"], methods[model_name]["model_config"]["config_name"], overrides=methods[model_name]["model_config"]["overrides"])

# Load a PDB file
pdb_data = load_pdb("lobster/model/latent_generator/example/example_pdbs/7kdr_protein.pdb")

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
from lobster.model.latent_generator.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.latent_generator.io import writepdb_ligand_complex, load_pdb, load_ligand 
import torch

model_name = 'LG Ligand 20A'

# Load model with ligand support
load_model(methods[model_name]["model_config"]["checkpoint"], methods[model_name]["model_config"]["config_path"], methods[model_name]["model_config"]["config_name"], overrides=methods[model_name]["model_config"]["overrides"])

# Load protein-ligand complex
pdb_data = {"protein_coords": None, "protein_mask": None, "protein_seq": None} 
ligand_data = load_ligand("latent_generator/example/example_pdbs/4erk_ligand.sdf")
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
  ligand_atoms=decoded_outputs["ligand_coords"][0],
  ligand_atom_names=None,  # Optional: provide atom names if available
  ligand_chain="L",
  ligand_resname="LIG")

```


### Protein-Ligand Complex Example
```python
from lobster.model.latent_generator.latent_generator.cmdline import load_model, encode, decode, methods
from lobster.model.latent_generator.latent_generator.io import writepdb_ligand_complex, load_pdb, load_ligand 
import torch

model_name = 'LG Ligand 20A seq 3di Aux'

# Load model with ligand support
load_model(methods[model_name]["model_config"]["checkpoint"], methods[model_name]["model_config"]["config_path"], methods[model_name]["model_config"]["config_name"], overrides=methods[model_name]["model_config"]["overrides"])

# Load protein-ligand complex
pdb_data = load_pdb("latent_generator/example/example_pdbs/4erk_protein.pdb")  
ligand_data = load_ligand("latent_generator/example/example_pdbs/4erk_ligand.sdf")
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
python latent_generator/cmdline/inference.py \
    --model_name 'LG 20A seq 3di c6d Aux' \
    --pdb_path latent_generator/example/example_pdbs/7kdr_protein.pdb \
    --decode

# Get tokens and decode to structure for ligand
python latent_generator/cmdline/inference.py \
    --model_name 'LG Ligand 20A' \
    --ligand_path latent_generator/example/example_pdbs/4erk_ligand.sdf  \
    --decode
    
# Get tokens and decode to structure for protein-ligand
python latent_generator/cmdline/inference.py \
    --model_name 'LG Ligand 20A seq 3di Aux' \
    --pdb_path latent_generator/example/example_pdbs/4erk_protein.pdb \
    --ligand_path latent_generator/example/example_pdbs/4erk_ligand.sdf  \
    --decode

# Get embeddings (requires Python API)
```

The tokens are discrete representations that can be used for tasks like discrete generation (with LLMs or PLMs) and compact storage of structure information, while embeddings are continuous representations useful for tasks like similarity search, feature extraction, and representation centric tasks.

For more detailed examples and use cases, see `latent_generator/example/tokenize_pdb_example.ipynb`.

## Training

### Protein-only Training
To train on protein structures only, use the default datamodule config with these recommended settings:
```bash
export HYDRA_FULL_ERROR=1

latent_generator_train \
    datamodule=structure_pinder_3di \
    datamodule.testing=false \
    +tokenizer.structure_encoder.spatial_attention_mask=true \
    tokenizer.optim.lr=1e-4 \
    +tokenizer.structure_encoder.angstrom_cutoff=20.0 \
    +tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0 \
    +tokenizer.structure_encoder.dropout=0.1 \
    +tokenizer.structure_encoder.attention_dropout=0.1 \
    tokenizer.structure_encoder.embed_dim=256 \
    tokenizer.quantizer.embed_dim=256 \
    tokenizer/decoder_factory=struc_decoder_3di_sequence \
    tokenizer/loss_factory=structure_losses_3di_sequence \
    +tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1 \
    +tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1 \
    +datamodule.use_shards=false \
    +trainer.log_every_n_steps=50 \
    +trainer.num_sanity_val_steps=0
```

Key settings explained:
- `datamodule=structure_pinder_3di`: Uses the 3Di-aware datamodule
- `tokenizer.structure_encoder.embed_dim=256`: 256-dimensional embeddings
- `tokenizer/decoder_factory=struc_decoder_3di_sequence`: Uses 3Di, aa sequence, and structure decoder
- `tokenizer/loss_factory=structure_losses_3di_sequence`: Uses 3Di, aa sequence, and structure sequence-aware loss

### Protein+Ligand (Complex) Training
To train on protein-ligand complexes, use the ligand datamodule config with these recommended settings:
```bash
export HYDRA_FULL_ERROR=1

latent_generator_train \
    datamodule=structure_ligand \
    datamodule.testing=false \
    datamodule.batch_size=20 \
    +tokenizer.structure_encoder.spatial_attention_mask=true \
    tokenizer.optim.lr=1e-4 \
    +tokenizer.structure_encoder.angstrom_cutoff=20.0 \
    +tokenizer.structure_encoder.angstrom_cutoff_spatial=20.0 \
    +tokenizer.structure_encoder.dropout=0.1 \
    +tokenizer.structure_encoder.attention_dropout=0.1 \
    tokenizer.structure_encoder.embed_dim=256 \
    +tokenizer.structure_encoder.encode_ligand=true \
    tokenizer/quantizer=slq_quantizer_ligand \
    tokenizer/decoder_factory=struc_decoder_3di_sequence \
    tokenizer/loss_factory=structure_losses_3di_sequence \
    +tokenizer.decoder_factory.decoder_mapping.vit_decoder.encode_ligand=true \
    +tokenizer.decoder_factory.decoder_mapping.vit_decoder.dropout=0.1 \
    +tokenizer.decoder_factory.decoder_mapping.vit_decoder.attention_dropout=0.1 \
    +datamodule.use_shards=false \
    +trainer.log_every_n_steps=50 \
    +trainer.num_sanity_val_steps=0
```

Key settings explained:
- `datamodule=structure_ligand`: Uses the ligand-aware datamodule
- `tokenizer.structure_encoder.embed_dim=256`: 256-dimensional embeddings
- `tokenizer.structure_encoder.encode_ligand=true`: Enables ligand encoding
- `tokenizer/quantizer=slq_quantizer_ligand`: Uses ligand-aware quantizer
- `tokenizer/decoder_factory=struc_decoder_3di_sequence`: Uses 3Di, aa sequence, and structure decoder
- `tokenizer/loss_factory=structure_losses_3di_sequence`: Uses 3Di, aa sequence, and structure sequence-aware loss

Make sure your dataset directory contains paired `*_protein.pt` and `*_ligand.pt` files for each complex.

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

#### LG 20A seq Aux
- **Description**: Sequence-aware protein model
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence decoder
  - 256 protein tokens
- **Use Case**: Protein structure analysis with sequence awareness

#### LG 20A seq 3di Aux
- **Description**: Sequence and 3Di-aware protein model
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di decoder
  - 256 protein tokens
- **Use Case**: Protein structure analysis with sequence and 3Di awareness

#### LG 20A seq 3di c6d Aux
- **Description**: Sequence, 3Di and C6D-aware protein model
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
- **Use Case**: Advanced protein structure analysis with sequence, 3Di and C6D features

#### LG 20A seq 3di c6d Aux Pinder
- **Description**: Sequence, 3Di and C6D-aware protein model (Pinder dataset)
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
- **Use Case**: Advanced protein structure analysis trained on Pinder dataset

#### LG 20A seq 3di c6d Aux PDB
- **Description**: Sequence, 3Di and C6D-aware protein model (PDB dataset)
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
- **Use Case**: Advanced protein structure analysis trained on PDB dataset

#### LG 20A seq 3di c6d Aux PDB Pinder
- **Description**: Sequence, 3Di and C6D-aware protein model (PDB + Pinder datasets)
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
- **Use Case**: Advanced protein structure analysis trained on combined PDB and Pinder datasets

#### LG 20A seq 3di c6d Aux PDB Pinder Finetune
- **Description**: Sequence, 3Di and C6D-aware protein model (finetuned on PDB + Pinder)
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
- **Use Case**: Finetuned protein structure analysis with sequence, 3Di and C6D features

#### LG 20A seq 3di c6d Aux dec960 PDB
- **Description**: Sequence, 3Di and C6D-aware protein model with 960-dim decoder
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
  - Decoder hidden dimension: 960
- **Use Case**: High-capacity protein structure analysis with sequence, 3Di and C6D features

#### LG 20A seq 3di c6d Aux dec960 PDB Finetune
- **Description**: Sequence, 3Di and C6D-aware protein model with 960-dim decoder (finetuned)
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
  - Decoder hidden dimension: 960
- **Use Case**: Finetuned high-capacity protein structure analysis

#### LG 20A seq 3di c6d 512 Aux
- **Description**: Sequence, 3Di and C6D-aware protein model with 512-dim embeddings
- **Features**:
  - 512-dim embeddings
  - 20Å spatial attention
  - Sequence + 3Di + C6D decoder
  - 256 protein tokens
- **Use Case**: High-dimensional protein structure analysis with sequence, 3Di and C6D features

#### LG 20A 3di c6d Aux
- **Description**: 3Di and C6D-aware protein model
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - 3Di + C6D decoder
  - 256 protein tokens
- **Use Case**: Advanced protein structure analysis with 3Di and C6D features

#### LG 20A c6d Aux
- **Description**: C6D-aware protein model
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - C6D decoder
  - 256 protein tokens
- **Use Case**: Protein structure analysis with C6D features

#### LG 20A 3di Aux
- **Description**: 3Di-aware protein model
- **Features**:
  - 256-dim embeddings
  - 20Å spatial attention
  - 3Di decoder
  - 256 protein tokens
- **Use Case**: Protein structure analysis with 3Di features

#### LG 20A
- **Description**: Basic protein model with 20Å cutoff
- **Features**:
  - Standard configuration
  - 20Å spatial attention
  - 256 protein tokens
- **Use Case**: Basic protein structure analysis

#### LG 10A
- **Description**: Basic protein model with 10Å cutoff
- **Features**:
  - Standard configuration
  - 10Å spatial attention
  - 256 protein tokens
- **Use Case**: Local protein structure analysis

#### LG full attention
- **Description**: Full attention model without spatial masking
- **Features**:
  - Standard configuration
  - Full attention (no spatial masking)
  - 256 protein tokens
- **Use Case**: Global protein structure analysis

To use any of these models, simply specify the model name when loading as keys the for the methods dictionary:
```python
from latent_generator.cmdline import load_model, methods

# Load a pre-configured model
model_name = 'LG 20A 3di c6d Aux'
load_model(
    methods[model_name]["model_config"]["checkpoint"],
    methods[model_name]["model_config"]["config_path"],
    methods[model_name]["model_config"]["config_name"],
    overrides=methods[model_name]["model_config"]["overrides"]
)
```

Or via command line:
```bash
python latent_generator/cmdline/inference.py --model_name 'LG 20A 3di c6d Aux' --pdb_path your_protein.pdb
```