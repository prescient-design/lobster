# Reinforcement Learning Training with UME

This guide demonstrates how to train language models using reinforcement learning with UME-based reward functions. The process involves two main steps: generating a synthetic dataset and then training with UME rewards.

### Package Structure

```
src/lobster/rl_training/
├── __init__.py              # Module exports
├── reward_functions.py      # UME reward functions and utilities
├── trainers.py             # Trainer creation and configuration
└── README.md              # Module documentation

tests/lobster/rl_training/
├── __init__.py
├── test_reward_functions.py # Unit tests for reward functions
└── test_trainers.py        # Unit tests for trainers

examples/
├── generate_synthetic_dataset.py  # Dataset generation script
├── train_ume_grpo.py             # Training script
└── test_ume_reward.py            # Testing script for reward functions
```

## Step-by-Step Training Process

## Prerequisites

Before starting the training process, ensure you have lobster installed with the TRL extra:

```bash
# Install lobster with all extras
uv sync --all-extras

# Or install lobster with TRL extra using pip
pip install "lbster[trl]"

# Or if installing from source
pip install -e ".[trl]"

# Alternative using uv (recommended for development)
uv pip install -e ".[trl]"
```

The TRL extra includes:
- `trl` - Transformer Reinforcement Learning library
- `accelerate` - HuggingFace Accelerate for distributed training

## Step 0: Download Qwen Model from Hugging Face Hub

Before running the training, you need to download a Qwen model checkpoint. The training script will automatically find and use Qwen models from your Hugging Face cache.

### Set up Hugging Face Cache Directory

First, set the `HF_HUB_CACHE` environment variable to specify where models should be cached:

```bash
# Set cache directory (adjust path as needed)
export HF_HUB_CACHE="/path/to/your/hf_cache"

# Or add to your shell profile for persistence
echo 'export HF_HUB_CACHE="/path/to/your/hf_cache"' >> ~/.bashrc
source ~/.bashrc
```

### Download Qwen Model

Download a Qwen model using the Hugging Face Hub CLI or Python:

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen2-0.5B-Instruct --local-dir-use-symlinks False

# Or using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2-0.5B-Instruct',
    local_dir_use_symlinks=False
)
"
```

### Alternative: Download via Python Script

Create a simple download script:

```python
#!/usr/bin/env python3
"""Download Qwen model for RL training."""

import os
from huggingface_hub import snapshot_download

def download_qwen_model():
    """Download Qwen model to HF_HUB_CACHE."""
    # Ensure HF_HUB_CACHE is set
    hf_cache = os.getenv('HF_HUB_CACHE')
    if not hf_cache:
        print("Warning: HF_HUB_CACHE not set. Using default cache location.")
    
    print("Downloading Qwen2-0.5B-Instruct model...")
    snapshot_download(
        repo_id='Qwen/Qwen2-0.5B-Instruct',
        local_dir_use_symlinks=False,
        cache_dir=hf_cache
    )
    print("Download completed!")

if __name__ == "__main__":
    download_qwen_model()
```

Save this as `download_qwen_model.py` and run:

```bash
python download_qwen_model.py
```

### Verify Model Download

Check that the model was downloaded correctly:

```bash
# List contents of cache directory
ls -la $HF_HUB_CACHE/models--Qwen--Qwen2-0.5B-Instruct/

# Should show something like:
# drwxr-xr-x 2 user user 4096 date snapshots/
# -rw-r--r-- 1 user user  123 date config.json
# -rw-r--r-- 1 user user  456 date README.md
# ...
```

The training script will automatically find and use this downloaded model.

### Step 1: Generate Synthetic Dataset

First, generate a synthetic dataset of molecular and biological sequences:

```bash
cd examples
python generate_synthetic_dataset.py
```

This script will:
- Generate 100 SMILES strings (molecular structures)
- Generate 100 amino acid sequences (proteins)
- Generate 100 DNA sequences
- Create train/validation/test splits (90%/5%/5%)
- Save the dataset to `synthetic_molecular_dataset/` directory

**Output:**
- `synthetic_molecular_dataset/` - HuggingFace dataset with train/val/test splits
- `synthetic_molecular_dataset.json` - JSON file for easy inspection

### Step 2: Run UME-based GRPO Training

After generating the dataset, run the training script:

```bash
python train_ume_grpo.py
```

This script will:
- Load the synthetic dataset from the generated directory
- Initialize a base language model (Qwen2-0.5B-Instruct)
- Load the UME model for reward computation
- Configure and start GRPO training with UME-based rewards
- Save training artifacts to the specified output directory

## Key Components

### 1. Reward Functions (`reward_functions.py`)

- **`UmeRewardFunction`**: Main reward function class that computes rewards based on UME pseudo-likelihood
- **`_detect_modality()`**: Automatically detects sequence modality (SMILES, amino acid, DNA) - available from `lobster.model.utils`
- **`compute_pseudo_likelihood()`**: Core function for computing likelihood scores
- **`create_ume_reward_wrapper()`**: Creates TRL-compatible wrapper functions

### 2. Trainers (`trainers.py`)

- **`create_ume_grpo_trainer()`**: Creates configured GRPO trainer with UME rewards
- **`train_ume_grpo()`**: Complete training pipeline (loads models, creates trainer, starts training)

### 3. Testing

- **Unit tests** in `tests/lobster/rl_training/` for proper test coverage
- **Integration tests** in `examples/test_ume_reward.py` for end-to-end testing

## Usage Examples

### Basic Training Pipeline

```python
# Step 1: Generate dataset
from examples.generate_synthetic_dataset import main as generate_dataset
generate_dataset()

# Step 2: Train with UME-based rewards
from lobster.rl_training import train_ume_grpo
from datasets import load_from_disk

# Load datasets
train_dataset = load_from_disk("synthetic_molecular_dataset/train")
val_dataset = load_from_disk("synthetic_molecular_dataset/validation")

# Train with UME-based rewards
trainer = train_ume_grpo(
    model_path="/path/to/base/model",
    ume_model_path="ume-mini-base-12M",
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir="./runs",
    reward_temperature=0.1,
    reward_batch_size=8,
)
```

### Custom Trainer Configuration

```python
from lobster.rl_training import create_ume_grpo_trainer
from lobster.model import Ume

# Load UME model
ume_model = Ume.from_pretrained("ume-mini-base-12M", device="cuda")

# Create custom trainer
trainer = create_ume_grpo_trainer(
    model_path="your/model",
    ume_model=ume_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir="./custom_runs",
    reward_temperature=0.2,
    reward_batch_size=16,
    learning_rate=1e-5,
    batch_size=4,
    max_steps=1000,
)
```

### Testing Reward Functions

```python
from lobster.rl_training import UmeRewardFunction
from lobster.model.utils import _detect_modality
from lobster.model import Ume

# Test modality detection
sequence = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
modality = _detect_modality(sequence)  # Returns Modality.SMILES

# Test reward function
ume_model = Ume.from_pretrained("ume-mini-base-12M", device="cuda")
reward_func = UmeRewardFunction(ume_model, temperature=0.1)
rewards = reward_func(["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"])
```

## Configuration Options

### Dataset Generation Parameters

The `generate_synthetic_dataset.py` script creates:
- **300 total sequences** (100 each of SMILES, amino acid, DNA)
- **Train/val/test split**: 90%/5%/5%
- **Sequence lengths**: 
  - SMILES: 10-50 characters
  - Amino acids: 20-100 characters  
  - DNA: 50-200 characters

### Reward Function Parameters

- **`temperature`**: Controls reward scaling (lower = more extreme rewards, default: 0.1)
- **`batch_size`**: Batch size for reward computation (default: 8)

### Training Parameters

- **`output_dir`**: Directory for training artifacts
- **`device`**: Device for UME model (default: "cuda")
- **Additional GRPO parameters**: All standard TRL GRPO parameters are supported

## Key Features

1. **Automatic Modality Detection**: Automatically detects and handles different sequence types
2. **Efficient Batching**: Processes sequences in batches for memory efficiency
3. **Temperature Scaling**: Configurable reward scaling for RL training
4. **Error Handling**: Robust error handling with fallback to zero rewards
5. **Modular Design**: Clean separation of concerns with reusable components
6. **Comprehensive Testing**: Unit tests and integration tests for reliability

## Files

- **`generate_synthetic_dataset.py`**: Creates synthetic molecular and biological sequences
- **`train_ume_grpo.py`**: Clean, minimal training script using the new modular structure
- **`test_ume_reward.py`**: Testing script for validating reward functions
- **Local model caching**: Qwen model is cached locally to avoid download timeouts

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `reward_batch_size` or use gradient accumulation
2. **Slow Training**: Increase `reward_batch_size` if memory allows
3. **Poor Rewards**: Adjust `reward_temperature` (lower = more extreme rewards)
4. **Dataset Not Found**: Ensure `generate_synthetic_dataset.py` was run first

### Performance Tips

- Use GPU for UME model inference
- Batch reward computations when possible
- Monitor memory usage during training
- Use appropriate sequence lengths for your use case