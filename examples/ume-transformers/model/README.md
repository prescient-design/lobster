# UME HuggingFace Transformers Deployment

This directory contains all files and documentation for deploying UME (Universal Molecular Encoder) as a standard HuggingFace model.

## 🎯 What We Achieved

**UME is now available as a standard HuggingFace model at**: https://huggingface.co/karina-zadorozhny/ume-base

Users can now use UME for multi-modal molecular understanding with just:
```bash
pip install transformers torch
```

No lobster installation required! ✨

## 📁 Files Overview

### 📖 Documentation
- **`DEPLOYMENT_GUIDE.md`** - Complete deployment process and technical details
- **`CONVERSION_SUMMARY.md`** - Technical conversion summary
- **`README.md`** - Hub repository documentation

### ⚙️ Configuration Files
- **`config.json`** - ModernBERT model configuration
- **`tokenizer_config.json`** - Tokenizer settings
- **`special_tokens_map.json`** - Special token mappings

### 🔧 Core Implementation
- **`tokenization_ume.py`** - Standalone multi-modal tokenizer (no lobster dependencies)

### 🧪 Testing & Scripts
- **`test_ume_hf.py`** - Comprehensive validation tests
- **`upload_to_hub.py`** - Hub upload script
- **`create_demo_weights.py`** - Demo weights generation
- **`demo_usage.py`** - End-user demonstration

## 🚀 Quick Usage

```python
from transformers import AutoModel, AutoTokenizer

# Load UME from HuggingFace Hub
model = AutoModel.from_pretrained("karina-zadorozhny/ume-base")
tokenizer = AutoTokenizer.from_pretrained("karina-zadorozhny/ume-base", trust_remote_code=True)

# Multi-modal molecular encoding
protein = "MKTVRQERLKSIVRILERSKEPVSGAQL"
inputs = tokenizer([protein], modality="amino_acid", return_tensors="pt")
outputs = model(**inputs)

print("Protein embedding shape:", outputs.last_hidden_state.shape)
```

## 🧬 Multi-Modal Support

UME handles three molecular modalities:

### Proteins (Amino Acids)
```python
protein = "MKTVRQERLKSIVRILERSKEPVSGAQL"
inputs = tokenizer([protein], modality="amino_acid", return_tensors="pt")
```

### Chemicals (SMILES)
```python
aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
inputs = tokenizer([aspirin], modality="SMILES", return_tensors="pt")
```

### DNA/RNA (Nucleotides)
```python
dna = "ATGCGTACGTAGCTACGTACG"
inputs = tokenizer([dna], modality="nucleotide", return_tensors="pt")
```

### Auto-Detection
```python
sequences = ["MKTVRQERLK", "CC(=O)O", "ATGCGT"]
for seq in sequences:
    inputs = tokenizer([seq], modality=None, return_tensors="pt")  # Auto-detect!
```

## 🏗️ Architecture

- **Base**: ModernBERT with RoPE + local-global attention
- **Parameters**: 14.2M (UME-mini)
- **Context**: 8,192 tokens
- **Vocabulary**: 100 unified tokens across modalities
- **Features**: Flash Attention, efficient long-context processing

## 🧪 Try It Out

1. **Run the demo**:
   ```bash
   python demo_usage.py
   ```

2. **Test the deployment**:
   ```bash
   python test_ume_hf.py
   ```

## 🎉 Impact

**First multi-modal molecular encoder as a standard HuggingFace model!**

This deployment makes UME accessible to the broader ML community:
- ✅ No specialized framework installation
- ✅ Standard transformers interface
- ✅ Multi-modal molecular understanding
- ✅ Research and production ready

## 📚 Learn More

- **Technical Details**: See `DEPLOYMENT_GUIDE.md`
- **Conversion Process**: See `CONVERSION_SUMMARY.md`
- **Hub Repository**: https://huggingface.co/karina-zadorozhny/ume-base
- **Live Demo**: Run `python demo_usage.py`

---

*This deployment demonstrates how specialized research models can be successfully adapted for broad community use while preserving their unique capabilities.* 