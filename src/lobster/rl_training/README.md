# Reinforcement Learning Training Module

This module provides utilities for training models using reinforcement learning techniques, particularly with reward functions based on UME models.

## Overview

The `rl_training` module contains:

- **Reward Functions**: UME-based reward functions for RL training
- **Trainers**: Utilities for creating and configuring RL trainers

## Quick Start

### Basic Usage

```python
from lobster.rl_training import train_ume_grpo
from datasets import load_from_disk

# Load your datasets
train_dataset = load_from_disk("path/to/train")
val_dataset = load_from_disk("path/to/val")

# Train with UME-based rewards
trainer = train_ume_grpo(
    model_path="your/base/model",
    ume_model_path="ume-mini-base-12M",
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir="./runs",
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

## Components

### Reward Functions (`reward_functions.py`)

- `UmeRewardFunction`: Main reward function class that computes rewards based on UME pseudo-likelihood
- `_detect_modality()`: Automatically detects sequence modality (SMILES, amino acid, DNA) - available from `lobster.model.utils`
- `compute_pseudo_likelihood()`: Core function for computing likelihood scores
- `create_ume_reward_wrapper()`: Creates TRL-compatible wrapper functions

### Trainers (`trainers.py`)

- `create_ume_grpo_trainer()`: Creates configured GRPO trainer with UME rewards
- `train_ume_grpo()`: Complete training pipeline (loads models, creates trainer, starts training)

## Configuration

### Reward Function Parameters

- `reward_temperature`: Controls reward scaling (lower = more extreme rewards, default: 0.1)
- `reward_batch_size`: Batch size for reward computation (default: 8)
- `penalty_for_invalid`: Penalty for invalid completions (default: -5.0)

### Penalty Value Considerations for GRPO Training

**Important**: The penalty value is crucial for GRPO training because GRPO normalizes rewards by standard deviation:

```
Âi,t = (ri - mean(r)) / std(r)
```

A penalty that's too mild may not provide sufficient negative signal. The default penalty of `-5.0` is suitable for most UME models, but you should verify this for your specific model.

**To determine the right penalty value:**

1. **Use the analysis script**:
   ```bash
   python examples/analyze_ume_rewards.py \
     --ume-model-path your/ume/model \
     --dataset-path your/dataset \
     --num-samples 100
   ```

2. **Manual calculation**: The penalty should be 2-3 standard deviations below the mean of valid rewards.

3. **Rule of thumb**: If your UME likelihoods are typically in range:
   - `[-10, 10]`: Use penalty `-5.0` to `-10.0`
   - `[-1, 1]`: Use penalty `-0.5` to `-1.0`  
   - `[0, 100]`: Use penalty `-50.0` to `-100.0`

### Training Parameters

- `output_dir`: Directory for training artifacts
- `device`: Device for UME model (default: "cuda")
- **Additional GRPO parameters**: All standard TRL GRPO parameters are supported

## Key Features

1. **Automatic Modality Detection**: Automatically detects and handles different sequence types
2. **Efficient Batching**: Processes sequences in batches for memory efficiency
3. **Temperature Scaling**: Configurable reward scaling for RL training
4. **Error Handling**: Robust error handling with fallback to zero rewards
5. **Modular Design**: Clean separation of concerns with reusable components

## Testing

Unit tests are located in `tests/lobster/rl_training/`:

- `test_reward_functions.py`: Tests for reward functions and modality detection
- `test_trainers.py`: Tests for trainer creation and configuration

For integration testing, see `examples/test_ume_reward.py`.

## Examples

See the `examples/` directory for complete working examples:

- `train_ume_grpo.py`: Clean, minimal training script using the modular structure
- `test_ume_reward.py`: Testing script for validating reward functions
- `README_RL_TRAINING.md`: Comprehensive documentation and usage examples 