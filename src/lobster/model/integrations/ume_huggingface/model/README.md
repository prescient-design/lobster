---
language:
- en
library_name: transformers
license: mit
tags:
- biology
- chemistry  
- molecular-modeling
- protein
- dna
- rna
- smiles
- drug-discovery
- multi-modal
- embeddings
- representation-learning
pipeline_tag: feature-extraction
base_model: modernbert
inference: true
model_type: modernbert
datasets:
- amplify
- zinc
- calm
- pdbind
- pinder
- atomica
tasks:
- feature-extraction
- text-classification
- text-similarity
- cross-modal-retrieval
widget:
- text: "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"
  example_title: "Protein Sequence"
- text: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" 
  example_title: "SMILES (Ibuprofen)"
- text: "ATGAAATTAGTTTAAGTCGCCAGAAGTAGTGAAAGGTGGTTAA"
  example_title: "DNA Sequence"
---

# Universal Molecular Encoder (UME)

UME is a foundation model that learns unified representations across protein sequences, DNA/RNA sequences, and small molecules (SMILES). It can embed different molecular modalities into a shared latent space, enabling cross-modal search, binding prediction, and molecular property prediction.

## Quick Start

```python
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

# Load model and config
config = AutoConfig.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True)
model = AutoModel.from_pretrained("karina-zadorozhny/ume", trust_remote_code=True, config=config)
```

#### Amino Acid Sequences (Proteins/Peptides)

```python
# Load amino acid tokenizer
tokenizer_aa = AutoTokenizer.from_pretrained(
    "karina-zadorozhny/ume", 
    subfolder="tokenizer_amino_acid",
    trust_remote_code=True
)

# Example protein sequences
sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA",
    "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
]

# Tokenize
inputs = tokenizer_aa(sequences, return_tensors="pt", padding=True, truncation=True)

# Get embeddings
with torch.no_grad():
    embeddings = model(inputs["input_ids"].unsqueeze(1), inputs["attention_mask"].unsqueeze(1))

print(f"Protein embeddings shape: {embeddings.shape}")
```

#### SMILES (Small Molecules)

```python
# Load SMILES tokenizer
tokenizer_smiles = AutoTokenizer.from_pretrained(
    "karina-zadorozhny/ume", 
    subfolder="tokenizer_smiles",
    trust_remote_code=True
)

# Example SMILES strings
smiles = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CC1=CC=CC=C1C(=O)O"               # Toluic acid
]

# Tokenize and embed
inputs = tokenizer_smiles(smiles, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    embeddings = model(inputs["input_ids"].unsqueeze(1), inputs["attention_mask"].unsqueeze(1))

print(f"SMILES embeddings shape: {embeddings.shape}")
```

#### Nucleotide Sequences (DNA/RNA)

```python
# Load nucleotide tokenizer
tokenizer_nucleotide = AutoTokenizer.from_pretrained(
    "karina-zadorozhny/ume", 
    subfolder="tokenizer_nucleotide",
    trust_remote_code=True
)

# Example DNA sequences
dna_sequences = [
    "ATGAAATTAGTTTAAGTCGCCAGAAGTAGTGAAAGGTGGTTAA",
    "ATGGCAATTGAAGAACCCGGTGGCATCGATGAAGTT"
]

# Tokenize and embed
inputs = tokenizer_nucleotide(dna_sequences, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    embeddings = model(inputs["input_ids"].unsqueeze(1), inputs["attention_mask"].unsqueeze(1))

print(f"DNA embeddings shape: {embeddings.shape}")
```


## Model Variants

| Model | Parameters | Layers | Hidden Size | Use Case |
|-------|------------|--------|-------------|----------|
| UME-mini | 12M | 6 | 384 | Fast inference, prototyping |
| UME-small | 90M | 12 | 768 | Balanced performance/speed |
| UME-medium | 480M | 24 | 1280 | High performance applications |
| UME-large | 740M | 24 | 1600 | Best performance |

Currently, all variants use the same model identifier. The default loaded model is UME-mini.

## Tokenizers

UME uses three specialized tokenizers for different molecular modalities:

- **`tokenizer_amino_acid`**: For protein and peptide sequences (20 standard amino acids + special tokens)
- **`tokenizer_smiles`**: For small molecules using chemical grammar-aware tokenization
- **`tokenizer_nucleotide`**: For DNA/RNA sequences using single nucleotide tokenization

## Key Features

- **Multi-modal**: Handles proteins, DNA/RNA, and small molecules in one model
- **Shared latent space**: Enables cross-modal similarity and search
- **Flexible context**: Supports sequences up to 8,192 tokens
- **Pre-trained**: Trained on ~1B molecular sequences across modalities

## Use Cases

- **Cross-modal search**: Find similar molecules across different representations
- **Binding prediction**: Predict protein-ligand interactions
- **Property prediction**: Molecular property prediction with transfer learning
- **Similarity search**: Find structurally or functionally similar molecules
- **Drug discovery**: Identify potential drug candidates

## Training Data

UME was trained on diverse molecular datasets including:
- **Proteins**: AMPLIFY, PeptideAtlas datasets
- **Small molecules**: ZINC, M³-20M datasets  
- **Nucleotides**: CaLM dataset
- **Interactions**: PINDER, PDBBind, ATOMICA datasets

## Limitations

- Nucleotide sequence performance is currently limited compared to protein and SMILES
- Context window of current 512 token (to be extended) may be restrictive for long genomic sequences
- Model performance varies by molecular modality and specific tasks

## Citation

```bibtex
@article{ume2024,
  title={Universal Molecular Encoder: Towards a Unified Representation of the Contents of the Cell},
  author = {Zadorozhny, Karina and Lisanza, Sidney and Joren, Taylor and Chennakesavalu, Shriram and Grambow, Colin and Kleinhenz, Joseph and Southern, Joshua and Choi, Keunwoo and Bonneau, Richard and Dwyer, Henri and Cho, Kyunghyun and Ra, Stephen and Frey, Nathan C.},
  year={2025},
  journal={Prescient Design, Genentech}
}
```

## License

This model is released under [LICENSE_TYPE]. Please see the repository for full license details.

## Links

- **Code**: [LOBSTER GitHub Repository](https://github.com/prescient-design/lobster)
- **Paper**: [Technical Report](link-to-paper)
- **Issues**: [GitHub Issues](https://github.com/prescient-design/lobster/issues)
