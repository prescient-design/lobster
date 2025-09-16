# Fine-tuning Module

This module provides fine-tuning functionality for UME (Universal Molecular Encoder) models, focusing on supervised learning tasks for molecular property prediction, retrieval, clustering, and other sequence analysis tasks.

## Overview

The fine-tuning module is designed to enhance UME model capabilities for downstream tasks such as:

- **Property Prediction**: Molecular Affinity (MA), Expression levels, and other molecular properties
- **Sequence Ranking**: Ranking molecular sequences by relevance or similarity
- **Retrieval**: Finding homologous sequences in large databases
- **Clustering**: Grouping similar molecular sequences

## Module Structure

```
src/lobster/post_train/
├── README.md                    # This file
├── __init__.py                  # Module exports
├── algorithms/                  # Training algorithms
│   ├── __init__.py
│   └── sft.py                  # Supervised fine-tuning
├── losses/                      # Loss functions
│   ├── __init__.py
│   ├── regression.py           # Regression losses
│   └── classification.py       # Classification losses
└── unfreezing.py               # Layer unfreezing strategies
```

## Components

### Algorithms (`algorithms/`)

#### Supervised Fine-tuning (`sft.py`)
- **`SupervisedFinetune`**: A [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) module for supervised fine-tuning of UME models
- Supports multiple task heads for multi-task learning
- Configurable optimizer and scheduler settings
- Integration with unfreezing strategies for gradual parameter updates
- Inspired by [xtuner](https://github.com/InternLM/xtuner)'s architecture with Lightning module structure

### Loss Functions (`losses/`)

#### Regression Losses (`regression.py`)
Advanced regression loss functions with sophisticated enhancements:

- **`MSELossWithSmoothing`**: Mean Squared Error with optional label smoothing
  - Supports Gaussian noise smoothing and Cortex-style moment averaging
  - Label smoothing techniques from [Szegedy et al. (2016)](https://arxiv.org/abs/1512.00567)
  
- **`HuberLossWithSmoothing`**: Robust Huber loss with label smoothing
  - More robust to outliers than MSE
  
- **`SmoothL1LossWithSmoothing`**: Smooth L1 loss (equivalent to Huber with delta=1.0)
  
- **`ExponentialParameterizedLoss`**: For properties spanning multiple orders of magnitude
  - Useful for molecular properties with exponential distributions
  
- **`NaturalGaussianLoss`**: Uncertainty-aware regression loss
  - Predicts both mean and variance for uncertainty quantification
  - Natural Gaussian parameterization inspired by [Cortex](https://github.com/prescient-design/cortex)
  - Essential for molecular property prediction where uncertainty matters

#### Classification Losses (`classification.py`)
Classification loss functions with class imbalance handling:

**Note**: For standard cross-entropy and binary cross-entropy, use PyTorch's built-in functions:
- `torch.nn.CrossEntropyLoss` (supports `label_smoothing`, `weight`, `reduction`)
- `torch.nn.functional.binary_cross_entropy_with_logits` (supports `pos_weight`, `reduction`)

- **`FocalLoss`**: Addresses class imbalance by focusing on hard examples and hard tokens (e.g. Ab HCDR3s)
  - Based on [Lin et al. (2017) "Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002)
  - Particularly useful for highly imbalanced molecular classification tasks
  - Not available in PyTorch's standard losses

### Unfreezing Strategies (`unfreezing.py`)

Sophisticated strategies for gradually unfreezing model parameters during fine-tuning:

- **`apply_unfreezing_strategy()`**: Apply various unfreezing strategies
  - `"full"`: Unfreeze all parameters immediately
  - `"partial"`: Unfreeze only the last N layers
  - `"progressive"`: Gradually unfreeze layers during training
  
- **`get_layer_wise_parameter_groups()`**: Layer-wise learning rate decay (LLRD)
  - Earlier layers get lower learning rates than later layers
  - Based on [ULMFiT approach (Howard & Ruder, 2018)](https://arxiv.org/abs/1801.06146)
  
- **`progressive_unfreezing_schedule()`**: Epoch-based progressive unfreezing
  - Unfreeze additional layers at specified training epochs
  - Prevents catastrophic forgetting during fine-tuning

## Key Features

### Advanced Label Smoothing
- **Gaussian Noise**: Simple additive noise for regularization
- **Moment Averaging**: Sophisticated smoothing based on prediction statistics ([Cortex](https://github.com/prescient-design/cortex)-inspired)
- **Dynamic Smoothing**: Adaptive smoothing based on model confidence

### Uncertainty Quantification
- **Natural Gaussian Loss**: Predicts both mean and variance
- **Uncertainty Metrics**: Standard deviation and negative log-likelihood tracking
- **Bayesian Approaches**: Inspired by [adaptive approximate inference methods](https://www.cs.toronto.edu/~cmaddis/pubs/aais.pdf)

### Progressive Training
- **Layer-wise Learning Rates**: Different learning rates for different model depths
- **Progressive Unfreezing**: Gradual parameter unfreezing to prevent catastrophic forgetting
- **Flexible Scheduling**: Customizable unfreezing schedules based on training epochs

### Multi-task Learning
- **Multiple Task Heads**: Support for simultaneous training on multiple tasks
- **Task-specific Losses**: Different loss functions for different tasks
- **Shared Representations**: Efficient parameter sharing across related tasks
- **Architecture**: Based on [Ruder (2017) "An Overview of Multi-Task Learning"](https://arxiv.org/abs/1706.05098)

## Usage Examples

### Basic Supervised Fine-tuning

```python
from lobster.model import UME
from lobster.post_train.algorithms import SupervisedFinetune
import torch.nn as nn

# Load pre-trained UME model
ume_model = UME(model_name="UME_mini")

# Define task heads
task_heads = {
    "affinity": nn.Linear(ume_model.embedding_dim, 1),  # Regression
    "toxicity": nn.Linear(ume_model.embedding_dim, 2),  # Binary classification
}

# Create fine-tuning model (uses default MSE loss for regression tasks)
post_train_model = SupervisedFinetune(
    ume_model=ume_model,
    task_heads=task_heads,
    unfreezing_strategy="progressive",
    optimizer_config={"lr": 2e-4, "weight_decay": 0.01}
)
```

### Advanced Loss Configuration

```python
# Method 1: Easy configuration using helper function
task_losses = SupervisedFinetune.create_loss_config(
    task_names=["affinity", "toxicity", "stability"],
    loss_types={
        "affinity": "natural_gaussian",  # Uncertainty-aware regression
        "toxicity": "focal",             # Handle class imbalance
        "stability": "huber"             # Robust to outliers
    },
    loss_kwargs={
        "affinity": {"label_smoothing": 0.1},
        "toxicity": {"gamma": 2.0, "alpha": 0.25},
        "stability": {"delta": 0.5, "label_smoothing": 0.05}
    }
)

# Method 2: Manual configuration
from lobster.model.losses import NaturalGaussianLoss, FocalLoss, HuberLossWithSmoothing
import torch.nn as nn

task_losses = {
    "affinity": NaturalGaussianLoss(label_smoothing=0.1),
    "toxicity": FocalLoss(gamma=2.0, alpha=0.25),  # Custom loss for class imbalance
    "stability": HuberLossWithSmoothing(delta=0.5, label_smoothing=0.05)
}

# Or use PyTorch's built-in losses directly:
task_losses_pytorch = {
    "affinity": NaturalGaussianLoss(label_smoothing=0.1),
    "toxicity": nn.CrossEntropyLoss(label_smoothing=0.05),  # PyTorch built-in
    "stability": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))  # For binary tasks
}

# Define task heads (note: NaturalGaussianLoss needs 2 outputs)
task_heads = {
    "affinity": nn.Linear(ume_model.embedding_dim, 2),   # mean + log_scale
    "toxicity": nn.Linear(ume_model.embedding_dim, 2),   # binary classification
    "stability": nn.Linear(ume_model.embedding_dim, 1)   # regression
}

# Create model with custom losses
post_train_model = SupervisedFinetune(
    ume_model=ume_model,
    task_heads=task_heads,
    task_losses=task_losses,
    task_types={"affinity": "regression", "toxicity": "classification", "stability": "regression"}
)
```

### Progressive Unfreezing

```python
from lobster.post_train.unfreezing import (
    apply_unfreezing_strategy,
    progressive_unfreezing_schedule,
    get_layer_wise_parameter_groups
)

# Apply initial unfreezing strategy
apply_unfreezing_strategy(model, "partial", num_layers=3)

# Set up layer-wise learning rates
param_groups = get_layer_wise_parameter_groups(
    model, 
    base_lr=2e-4, 
    decay_factor=0.9
)

# Progressive unfreezing during training
unfreeze_schedule = [5, 10, 15]  # Epochs to unfreeze additional layers
progressive_unfreezing_schedule(model, current_epoch=10, unfreeze_schedule=unfreeze_schedule)
```

## Integration with Existing Codebase

This fine-tuning module integrates seamlessly with the existing lobster ecosystem:

- **UME Models**: Direct compatibility with `lobster.model.UME`
- **Datasets**: Works with all existing `lobster.datasets` implementations
- **Training**: Compatible with existing `lobster.cmdline.train` infrastructure
- **Metrics**: Evaluation metrics available in `lobster.metrics`

## References and Inspirations

- **[xtuner](https://github.com/InternLM/xtuner)**: Architecture patterns and registry system
- **[Cortex](https://github.com/prescient-design/cortex)**: Advanced loss functions and label smoothing techniques
- **[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)**: Training loop and module structure
- **[ULMFiT](https://arxiv.org/abs/1801.06146)**: Progressive unfreezing and layer-wise learning rates
- **[Focal Loss](https://arxiv.org/abs/1708.02002)**: Class imbalance handling for molecular classification
- **[Label Smoothing](https://arxiv.org/abs/1512.00567)**: Regularization techniques for better generalization
- **[Bayesian Neural Networks](https://www.cs.toronto.edu/~cmaddis/pubs/aais.pdf)**: Uncertainty quantification in regression tasks

## Future Extensions

The modular design allows for easy extension with additional components:

- **Contrastive Learning**: InfoNCE and other contrastive objectives
- **PEFT Methods**: LoRA and adapter-based fine-tuning
- **Reinforcement Learning**: GRPO/PPO-style fine-tuning
- **Advanced Pooling**: Attention-based and learnable pooling strategies
- **Normalization**: Whitening and other embedding normalization techniques