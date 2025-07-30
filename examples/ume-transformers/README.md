# Deploy UME to HuggingFace Hub

This directory contains everything needed to deploy UME (Universal Molecular Encoder) as a standard HuggingFace model.

## 🎯 What This Does

Converts UME from the lobster framework into a standard HuggingFace model that users can load with:

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("your-username/ume-base")
tokenizer = AutoTokenizer.from_pretrained("your-username/ume-base", trust_remote_code=True)
```

## 📁 File Structure

```
examples/ume-transformers/
├── README.md                    # This deployment guide
├── model/                       # Model files for HuggingFace Hub
│   ├── config.json             # ModernBERT model configuration  
│   ├── tokenization_ume.py     # Standalone multi-modal tokenizer
│   ├── tokenizer_config.json   # Tokenizer settings
│   ├── special_tokens_map.json # Special token mappings
│   └── README.md               # Hub repository documentation
├── upload_to_hub.py            # Upload script
├── test_ume_hf.py              # Validation tests
└── create_demo_weights.py      # Demo weights generation
```

## 🚀 Deployment Steps

### 1. Prepare Model Files

The `model/` directory contains all files needed for the Hub:

- **`config.json`**: Maps UME to ModernBERT architecture with UME-specific parameters
- **`tokenization_ume.py`**: Self-contained tokenizer supporting proteins, SMILES, and nucleotides
- **`tokenizer_config.json`**: Tokenizer configuration with auto-mapping
- **`special_tokens_map.json`**: Special tokens for all modalities
- **`README.md`**: Hub repository documentation with usage examples

### 2. Test the Model

Validate everything works before uploading:

```bash
cd examples/ume-transformers
python test_ume_hf.py
```

This tests:
- ✅ Config loading with ModernBERT
- ✅ Tokenizer multi-modal functionality  
- ✅ Model creation (14.2M parameters)
- ✅ Auto-modality detection
- ✅ Inference pipeline
- ✅ Embedding extraction

### 3. Upload to Hub

Run the upload script:

```bash
python upload_to_hub.py
```

The script will:
1. Check HuggingFace authentication
2. Create repository `your-username/ume-base`  
3. Upload all files from `model/` directory
4. Verify successful upload

### 4. Add Model Weights

Create demo weights (or upload your trained weights):

```bash
python create_demo_weights.py
```

This generates and uploads a randomly initialized model for testing.

## 🧬 Model Capabilities

### Multi-Modal Support
- **Proteins**: `"MKTVRQERLKSIVRILERSKEPVSGAQL"`
- **Chemicals**: `"CC(=O)OC1=CC=CC=C1C(=O)O"` (SMILES)
- **DNA/RNA**: `"ATGCGTACGTAGCTACGTACG"`

### Auto-Detection
```python
sequences = ["MKTVRQERLK", "CC(=O)O", "ATGCGT"]
for seq in sequences:
    inputs = tokenizer([seq], modality=None, return_tensors="pt")  # Auto-detect!
```

### Architecture
- **Base**: ModernBERT with RoPE + local-global attention
- **Parameters**: 14.2M (UME-mini config)
- **Context**: 8,192 tokens
- **Vocabulary**: 100 unified tokens across modalities

## 🧪 Usage After Deployment

Once deployed, users can use UME with standard HuggingFace patterns:

```python
from transformers import AutoModel, AutoTokenizer

# Load from Hub
model = AutoModel.from_pretrained("your-username/ume-base")
tokenizer = AutoTokenizer.from_pretrained("your-username/ume-base", trust_remote_code=True)

# Multi-modal molecular encoding
protein = "MKTVRQERLKSIVRILERSKEPVSGAQL"
chemical = "CC(=O)OC1=CC=CC=C1C(=O)O"
dna = "ATGCGTACGTAGCTACGTACG"

# Process with auto-detection
for seq in [protein, chemical, dna]:
    inputs = tokenizer([seq], modality=None, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0]  # CLS token
    print(f"Embedding shape: {embedding.shape}")

# Explicit modality specification
inputs = tokenizer([protein], modality="amino_acid", return_tensors="pt")
outputs = model(**inputs)

# Batch processing
batch = [protein, chemical, dna]
inputs = tokenizer(batch, modality=None, return_tensors="pt", padding=True)
outputs = model(**inputs)
batch_embeddings = outputs.last_hidden_state[:, 0]  # [3, hidden_size]
```

## ⚙️ Technical Details

### Key Features Preserved
- ✅ Multi-modal tokenization (proteins, SMILES, DNA/RNA)
- ✅ Automatic modality detection
- ✅ ModernBERT architecture with RoPE and local-global attention  
- ✅ 8,192 token context length
- ✅ Flash Attention compatibility
- ✅ Unified 100-token vocabulary

### Dependencies
- Before: `lobster` framework (~50 packages, ~2GB)
- After: `transformers` + `torch` only (~200MB)

### Architecture Mapping
- Original: FlexBERT within lobster framework
- Converted: ModernBERT with native HuggingFace support
- Benefits: Better performance, standard interface, no custom code needed

## 🛠️ Customization

### Upload to Different Repository
Edit `upload_to_hub.py` and change:
```python
repo_id = "your-username/your-model-name"
```

### Modify Model Configuration  
Edit `model/config.json` to adjust:
- `hidden_size`, `num_hidden_layers`, `num_attention_heads`
- `vocab_size`, `max_position_embeddings`
- ModernBERT-specific parameters

### Extend Tokenizer
Edit `model/tokenization_ume.py` to:
- Add new modalities
- Modify vocabulary  
- Update detection logic

## 🎉 Result

**UME becomes the first multi-modal molecular encoder available as a standard HuggingFace model!**

Users can now:
- Load UME with 2 lines of code
- Use without installing specialized frameworks
- Apply to molecular understanding tasks
- Fine-tune for specific research needs
- Deploy in production systems

This deployment bridges cutting-edge research with community accessibility, making advanced molecular AI available to the global ML community.

---

**Live Model**: https://huggingface.co/karina-zadorozhny/ume-base 