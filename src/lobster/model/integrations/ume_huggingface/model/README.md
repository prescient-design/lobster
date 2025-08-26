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

UME is a **universal foundation model** that learns unified representations across the diverse languages of molecular biology. Built on ModernBERT and trained through a 3-stage curriculum, UME embeds proteins, small molecules, DNA/RNA sequences, and 3D structures into a shared, structure-aware latent space.

Unlike traditional models that focus on single molecular modalities, UME enables **cross-modal reasoning** including similarity search, representation translation, and multi-entity interaction modeling. UME scales along the **modality dimension** in addition to traditional scaling of model size, compute, and data - trained on ~1B molecular sequences with 170B tokens across diverse biological datasets.

UME is explicitly trained to encode molecular interactions (protein-ligand, protein-protein, protein-DNA/RNA complexes), enabling novel capabilities for computational biology and drug discovery applications.

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


## Architecture

UME is built on **ModernBERT**, an enhanced transformer architecture optimized for long-context processing and computational efficiency. Key architectural features include:

- **Long Context Support**: Supports sequences up to 8,192 tokens (currently trained on 512 tokens)
- **Enhanced Activations**: Uses GeGLU activations instead of GELU for greater expressiveness
- **Efficient Attention**: Alternating attention scheme with interleaved local and global heads
- **Memory Optimized**: Flash Attention and unpadding optimizations for reduced memory usage
- **Encoder-Only Design**: Processes full input sequences in parallel for bidirectional representations

The encoder-only architecture is specifically chosen for UME's objective of learning unified molecular representations, providing richer context for cross-modal alignment and producing fixed-length embeddings suitable for downstream tasks.

## Model Variants

| Model | Parameters | Layers | Hidden Size | Attention Heads | Use Case |
|-------|------------|--------|-------------|-----------------|----------|
| UME-mini | 12M | 6 | 384 | 6 | Fast inference, prototyping |
| UME-small | 90M | 12 | 768 | 12 | Balanced performance/speed |
| UME-medium | 480M | 24 | 1280 | 20 | High performance applications |
| UME-large | 740M | 24 | 1600 | 25 | Best performance |

All model sizes are optimized for GPU hardware efficiency following established best practices. Currently, all variants use the same model identifier. The default loaded model is UME-medium.

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
- **Interaction modeling**: Explicitly trained on molecular complexes and interactions
- **Structure-aware**: Integrates 3D structural information via LatentGenerator alignment
- **Curriculum learning**: Progressive training from simple to complex multi-modal objectives

## Understanding Biological Multi-Modality

UME addresses different types of biological multi-modality:

### Intra-Entity Modalities
Different representations of the **same biological entity**:
- Protein sequence → SMILES representation (chemical view of peptide)
- DNA sequence → Amino acid sequence (towards central dogma)
- Sequence → 3D structure (different views of same molecule)

### Inter-Entity Modalities  
Interactions between **distinct biological entities**:
- Protein-drug binding (therapeutic interactions)
- Protein-protein interactions (cellular complexes)
- Protein-DNA/RNA binding (gene regulation)

This distinction is crucial for UME's design: intra-entity tasks require learning alignment between different representations, while inter-entity tasks require modeling compatibility and emergent interaction properties.

## Use Cases

- **Cross-modal search**: Find similar molecules across different representations
- **Binding prediction**: Predict protein-ligand interactions
- **Property prediction**: Molecular property prediction with transfer learning
- **Similarity search**: Find structurally or functionally similar molecules
- **Drug discovery**: Identify potential drug candidates

## Training Methodology

UME employs a **3-stage curriculum learning** approach, progressively increasing complexity:

### Stage 1: Unimodal Language Modeling
- Standard masked language modeling (MLM) on individual sequence types
- 25% masking rate across amino acid, SMILES, and nucleotide sequences
- Builds robust internal representations for each modality independently

### Stage 2: Cross-Modal Alignment
- Combines MLM with contrastive learning (CLIP/Symile objectives)
- Aligns different representations of the same molecular entity
- Supports transformations like amino acid ↔ SMILES, nucleotide ↔ amino acid
- Integrates 3D structural information via LatentGenerator alignment

### Stage 3: Multi-Entity Interaction
- Models interactions between different molecular entities
- Includes protein-protein, protein-ligand, protein-DNA/RNA complexes
- Maintains replay buffer (20% previous data) to prevent catastrophic forgetting
- Incorporates structural information when available

## Training Data

UME was trained on **~1B molecular sequences** with **170B tokens** across diverse datasets:

| Dataset | Modality | Size | Description |
|---------|----------|------|-------------|
| **AMPLIFY** | Proteins | 360.7M sequences | UniRef100, antibody databases, structural classifications |
| **PeptideAtlas** | Peptides | 4.2M sequences | Experimentally validated human peptides from mass spectrometry |
| **ZINC** | SMILES | 588.7M compounds | Purchasable compounds filtered for drug-like properties |
| **M³-20M** | SMILES | 20.8M molecules | Multi-modal molecular dataset with annotations |
| **CaLM** | Nucleotides | 8.6M sequences | Protein-coding DNA sequences from European Nucleotide Archive |
| **PINDER** | Protein-Protein | 267K structures | Interaction structures with apo/holo states and predictions |
| **ATOMICA** | Multi-type | 360.7M | Diverse molecular interactions and structures |
| **GEOM** | 3D Structures | 1.17M conformers | Small molecule conformations for structural alignment |

### Cross-Modal Transformations
UME learns relationships between modalities through deterministic and probabilistic transformations:
- **Amino acid ↔ SMILES**: Chemical representation of peptides
- **Nucleotide ↔ Amino acid**: Central dogma of molecular biology  
- **Sequence ↔ 3D Structure**: Via LatentGenerator structural encoder
- **Canonical ↔ Randomized SMILES**: Chemical string invariance


## Frequently Asked Questions (FAQ)

### Model Architecture
**Q: What architecture is UME based on?**
- ModernBERT encoder-only transformer
- GeGLU activations, Flash Attention, alternating local/global attention

**Q: What are the model sizes?**
- UME-mini: 12M parameters, 6 layers, 384 hidden size
- UME-small: 90M parameters, 12 layers, 768 hidden size  
- UME-medium: 480M parameters, 24 layers, 1280 hidden size
- UME-large: 740M parameters, 24 layers, 1600 hidden size

**Q: What's the context length?**
- Trained on: 512 tokens
- Architecture supports: up to 8,192 tokens

### Training Data
**Q: How much data was UME trained on?**
- ~1B molecular sequences
- 170B tokens total
- 9 major datasets across modalities

**Q: What datasets were used?**
- **Proteins**: AMPLIFY (360.7M), PeptideAtlas (4.2M)
- **Small molecules**: ZINC (588.7M), M³-20M (20.8M)
- **Nucleotides**: CaLM (8.6M)
- **Structures**: PINDER (267K), GEOM (1.17M)

**Q: What training stages were used?**
- Stage 1: Masked language modeling on individual modalities
- Stage 2: Cross-modal alignment with contrastive learning
- Stage 3: Multi-entity interaction modeling

### Supported Modalities
**Q: What molecular types does UME handle?**
- Amino acid sequences (proteins/peptides)
- SMILES strings (small molecules)
- Nucleotide sequences (DNA/RNA)
- 3D structures (via LatentGenerator)

**Q: Which tokenizer should I use?**
- `tokenizer_amino_acid`: protein/peptide sequences
- `tokenizer_smiles`: small molecules
- `tokenizer_nucleotide`: DNA/RNA sequences

### Capabilities & Limitations
**Q: Can UME generate sequences?**
- Yes with some additional work (Gibbs Sampling, primarily for infilling/inpainting, conditional generation)
 
**Q: What are the main limitations?**
- Nucleotide performance limited vs proteins/SMILES
- 512-token training context may restrict long sequences
- 3D structures limited to proteins and small molecules

## Limitations

- **Nucleotide sequence performance** is currently limited compared to protein and SMILES representations, likely due to underrepresentation in training data and shorter context requirements for genomic sequences
- **Context window** of 512 tokens (during training) may be restrictive for long genomic sequences, though the architecture supports up to 8,192 tokens
- **Model performance varies** by molecular modality and specific tasks, with strongest performance on protein and small molecule tasks
- **Structural modeling** is currently limited to protein and protein-ligand complexes; DNA/RNA structures not yet supported
- **Dataset imbalance** leads to varying performance across modalities due to differences in training data volume

## Interactive Tools & Infrastructure

UME is designed with usability and reproducibility in mind:

- **Hugging Face Integration**: Pre-trained models available through the Hugging Face ecosystem
- **MCP Integration**: Model Control Protocol for AI-assisted research workflows  
- **Web Interface**: Interactive exploration of embedding spaces and cross-modal search
- **LOBSTER Framework**: Complete open-source codebase for training and evaluation

## Citation

```bibtex
@article{ume2025,
  title={Universal Molecular Encoder: Towards a Unified Representation of the Contents of the Cell},
  author={Zadorozhny, Karina and Lisanza, Sidney and Joren, Taylor and Chennakesavalu, Shriram and Grambow, Colin and Kleinhenz, Joseph and Southern, Joshua and Choi, Keunwoo and Bonneau, Richard and Dwyer, Henri and Cho, Kyunghyun and Ra, Stephen and Frey, Nathan C.},
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
