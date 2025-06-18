# Universal Molecular Encoder (UME)

The Universal Molecular Encoder (UME) is a powerful transformer-based model designed for encoding molecular data across multiple modalities including SMILES strings, amino acid sequences, and nucleotide sequences. It provides a unified interface for molecular representation learning with support for both training and inference.


## Available Models

As of today (pre-release, June 18, 2025), the following pretrained UME models are available:

| Model Name | Parameters |
|------------|------------|
| `ume-mini-base-12M` | 12M | 
| `ume-medium-base-480M` | 480M | 
| `ume-large-base-740M` | 740M | 

These models are currently just placeholder checkpoints and only available to member of Prescient Design.
 Stay tuned for public release.

## Quick Start

#### Loading a Pretrained Model

```python
from lobster.model import Ume

# Load UME-mini with automatic device detection
model = Ume.from_pretrained("ume-mini-base-12M")

# Load with specific device
model = Ume.from_pretrained("ume-mini-base-12M", device="cpu")

# Load with custom cache directory
model = Ume.from_pretrained("ume-mini-base-12M", cache_dir="/path/to/cache")
```

#### Getting Embeddings

```python
# Protein sequence embeddings
protein_sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
embeddings = model.embed_sequences(protein_sequences, "amino_acid")
print(embeddings.shape)  # torch.Size([1, 768])

# SMILES embeddings
smiles_strings = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"]  # Aspirin, Ibuprofen
embeddings = model.embed_sequences(smiles_strings, "SMILES")
print(embeddings.shape)  # torch.Size([2, 768])

# DNA sequence embeddings
dna_sequences = ["ATGCATTGCA", "GCTAGCTA"]
embeddings = model.embed_sequences(dna_sequences, "nucleotide")
print(embeddings.shape)  # torch.Size([2, 768])
```

#### Token-level Embeddings

```python
# Get token-level embeddings (without aggregation)
token_embeddings = model.embed_sequences(dna_sequences, "nucleotide", aggregate=False)
print(token_embeddings.shape)  # torch.Size([2, 10, 768]) - [batch_size, seq_len, hidden_dim]
```

## Advanced Usage

### Model Initialization

```python
from lobster.model import Ume

# Initialize a new model from scratch
encoder = Ume(
    model_name="UME_mini",
    max_length=512,
    lr=1e-3,
    mask_percentage=0.25
)

# Load from a custom checkpoint
encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt", device="cuda")
```

### Training Configuration

```python
# Initialize with contrastive learning
encoder = Ume(
    model_name="UME_mini",
    contrastive_loss_type="clip",  # or "symile", "disco_clip"
    contrastive_loss_weight=0.5,   # Balance between MLM and contrastive loss
    contrastive_temperature=0.07
)

# Configure learning rate scheduler
encoder = Ume(
    model_name="UME_mini",
    scheduler="constant_with_warmup",
    num_training_steps=10000,
    num_warmup_steps=1000
)
```

### Model Freezing/Unfreezing

```python
# Freeze model for inference
model.freeze()
print(f"Model is frozen: {model.frozen}")  # True

# Unfreeze for fine-tuning
model.unfreeze()
print(f"Model is frozen: {model.frozen}")  # False
```

### Direct Tokenization

```python
# Get tokenizer for specific modality
tokenizer = model.get_tokenizer("amino_acid")
sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQL"]
tokens = tokenizer(sequences, return_tensors="pt")
print(tokens.keys())  # dict_keys(['input_ids', 'attention_mask'])

# Get vocabulary
vocab = model.get_vocab()
print(f"Vocabulary size: {len(vocab)}")
```

### Low-level Embedding

```python
# Direct embedding from tokenized inputs
inputs = {
    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
}
embeddings = model.embed(inputs, aggregate=True)
print(embeddings.shape)  # torch.Size([1, 768])
```

## Supported Modalities

UME supports the following molecular modalities:

- **SMILES**: Chemical structure representation
- **amino_acid**: Protein sequences
- **nucleotide**: DNA/RNA sequences

You can check available modalities:

```python
print(model.modalities)  # ['SMILES', 'amino_acid', 'nucleotide', '3d_coordinates']
```

## Model Architecture

UME is built on top of FlexBERT with the following key features:

- **Flash Attention**: Optimized attention computation for GPU
- **Flexible Padding**: Support for both padded and unpadded architectures
- **Multi-modal Tokenization**: Specialized tokenizers for each modality
- **Contrastive Learning**: Support for various contrastive loss types

## Training

### Loss Types

UME supports multiple training objectives:

1. **Masked Language Modeling (MLM)**: Standard masked token prediction
2. **Contrastive Learning**: 
   - **CLIP**: Standard InfoNCE loss for 2 views
   - **Symile**: Multi-view contrastive loss for ≥2 views
   - **DiscoCLIP**: Memory-efficient distributed CLIP loss

## Updating checkpoints

UME checkpoints are managed through a dedicated command-line tool that allows you to list, add, update, and delete checkpoints in the S3 registry.

### Available Actions

The checkpoint manager supports the following actions:

- **`list`**: Display all available checkpoints
- **`add`**: Add a new checkpoint (fails if model already exists)
- **`update`**: Update an existing checkpoint
- **`delete`**: Remove a checkpoint from the registry

### Usage Examples

#### List All Checkpoints

```bash
uv run python src/lobster/cmdline/manage_ume_checkpoints.py action=list
```

This will display all currently available checkpoints with their S3 paths.

#### Add a New Checkpoint

```bash
uv run python src/lobster/cmdline/manage_ume_checkpoints.py \
    action=add \
    model_name=ume-small-base-90M \
    checkpoint_path=s3://prescient-lobster/ume/runs/2025-06-18T10-00-00/epoch=0-step=50000-val_loss=0.6500.ckpt \
    dry_run=true
```

**Note**: Use `dry_run=true` to preview changes without actually making them. Remove this parameter to apply the changes.

#### Update an Existing Checkpoint

```bash
uv run python src/lobster/cmdline/manage_ume_checkpoints.py \
    action=update \
    model_name=ume-mini-base-12M \
    checkpoint_path=s3://prescient-lobster/ume/runs/2025-06-18T15-30-00/epoch=0-step=5000-val_loss=0.7500.ckpt \
    dry_run=true
```

This will update the checkpoint path for an existing model.

#### Delete a Checkpoint

```bash
uv run python src/lobster/cmdline/manage_ume_checkpoints.py \
    action=delete \
    model_name=ume-small-base-90M-test \
    dry_run=true
```

This removes a checkpoint from the registry (the actual file in S3 is not deleted).

### Important Notes

1. **S3 Paths Only**: All checkpoint paths must be valid S3 URIs starting with `s3://`
2. **Dry Run Mode**: Always use `dry_run=true` first to preview changes
3. **Model Names**: Use consistent naming convention (e.g., `ume-{size}-base-{params}M`)
4. **Permissions**: Ensure you have appropriate S3 permissions to read/write the checkpoint registry

### Checkpoint Registry Location

The checkpoint registry is stored at: `s3://prescient-lobster/ume/checkpoints.json`

This file contains a JSON mapping of model names to their S3 checkpoint paths and is automatically managed by the checkpoint manager tool.

## Quick Start
