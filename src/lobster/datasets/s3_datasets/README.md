# S3 Lobster Datasets

This directory contains dataset classes for efficiently loading large-scale biological sequence data from S3 storage during training.

## Overview

For efficient training at scale, we avoid streaming directly from HuggingFace to prevent networking bottlenecks and reliability issues. Instead, each dataset has been:

1. **Downloaded from HuggingFace** and uploaded to our S3 bucket
2. **Optimized using `litdata.optimize`** for maximum performance during training

## Available Datasets

- **AMPLIFY** - Amino acid sequences (360M+ training samples)
- **Atomica** - Multi-modal sequence pairs (309K+ training samples)  
- **ZINC** - SMILES molecular sequences (588M+ training samples)
- **PeptideAtlas** - Amino acid peptide sequences (79M+ training samples)
- **Calm** - Nucleotide sequences (8.6M+ training samples)
- **M320M** - SMILES molecular sequences (20M+ training samples)

## Usage

Each dataset class provides both standard and lightning-optimized splits:

```python
from lobster.datasets.s3_datasets import AMPLIFY
from lobster.constants import Split

# Standard parquet files
dataset = AMPLIFY(split=Split.TRAIN, max_length=512)

# Lightning-optimized files (recommended for large-scale training)
dataset = AMPLIFY(split=Split.TRAIN, max_length=512, use_optimized=True)
```

## Optimization

All datasets support lightning-optimized versions created with `litdata.optimize`. See:
- `_optimize.py` for optimization scripts
- [LitData documentation](https://github.com/Lightning-AI/litData?tab=readme-ov-file#option-2-optimize-for-maximum-performance) for details

**Recommendation**: Use `use_optimized=True` when training large-scale models like UME for best performance.