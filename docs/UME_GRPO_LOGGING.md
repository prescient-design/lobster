# UME GRPO Training with Enhanced Wandb Logging

This document describes the enhanced logging and monitoring capabilities for UME-based GRPO training, including detailed sample tracking, reward analysis, and modality-specific metrics.

## Overview

The enhanced logging system provides comprehensive monitoring of UME GRPO training through:

1. **UmeGrpoLoggingCallback**: A custom Lightning callback that logs training samples and rewards
2. **Enhanced Reward Function**: Detailed reward computation logging with modality breakdown
3. **Wandb Integration**: Automatic wandb run management and metric logging

## Features

### 1. Training Sample Logging

The system logs individual training samples with their:
- **Completions**: The generated sequences (truncated for display)
- **Rewards**: UME-based reward scores
- **Modalities**: Detected sequence type (SMILES, amino acid, nucleotide)
- **Length**: Sequence length

### 2. Reward Statistics

Comprehensive reward tracking including:
- **Basic Statistics**: Mean, min, max, standard deviation
- **Modality Breakdown**: Separate statistics for each modality type
- **Histograms**: Distribution of rewards over time
- **Cumulative Metrics**: Total samples processed, overall statistics

### 3. Training Progress Monitoring

Real-time monitoring of:
- **Step-level metrics**: Per-batch reward statistics
- **Epoch-level summaries**: Aggregated metrics per epoch
- **Training configuration**: Model paths, hyperparameters, etc.

## Usage

### Basic Usage

```python
from lobster.rl_training import train_ume_grpo

# Start training with enhanced logging
trainer = train_ume_grpo(
    model_path="your/base/model",
    ume_model_path="ume-mini-base-12M",
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    enable_wandb_logging=True,
    wandb_project="your-project",
    wandb_entity="your-username",
    wandb_run_name="experiment-name",
)
```

### Advanced Configuration

```python
from lobster.rl_training import create_ume_grpo_trainer
from lobster.callbacks import UmeGrpoLoggingCallback

# Create custom logging callback
logging_callback = UmeGrpoLoggingCallback(
    log_every_n_steps=50,           # Log every 50 steps
    max_samples_to_log=15,          # Log up to 15 samples per step
    log_reward_histogram=True,      # Enable reward histograms
    log_sample_examples=True,       # Enable sample logging
    log_modality_breakdown=True,    # Enable modality breakdown
)

# Create trainer with custom callback
trainer = create_ume_grpo_trainer(
    model_path="your/base/model",
    ume_model=ume_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    enable_wandb_logging=True,
    callbacks=[logging_callback],
)
```

### Environment Variables

Set these environment variables for wandb configuration:

```bash
export WANDB_ENTITY="your-username"
export WANDB_PROJECT="your-project"
export WANDB_API_KEY="your-api-key"
```

## Wandb Dashboard

The logging system creates a comprehensive wandb dashboard with:

### 1. Training Metrics

- **train/mean_reward**: Average reward per batch
- **train/min_reward**: Minimum reward in batch
- **train/max_reward**: Maximum reward in batch
- **train/reward_std**: Standard deviation of rewards
- **train/batch_size**: Number of samples per batch

### 2. Modality-Specific Metrics

- **train/modality_smiles/mean_reward**: Average reward for SMILES sequences
- **train/modality_amino_acid/mean_reward**: Average reward for amino acid sequences
- **train/modality_nucleotide/mean_reward**: Average reward for nucleotide sequences
- **train/modality_*/count**: Number of samples per modality

### 3. Sample Examples

Interactive tables showing:
- Sample completions (truncated)
- Individual reward scores
- Modality classification
- Sequence length

### 4. Reward Histograms

Distribution plots showing:
- Overall reward distribution
- Modality-specific distributions
- Evolution over training steps

### 5. Epoch Summaries

- **epoch/mean_reward**: Average reward per epoch
- **epoch/min_reward**: Minimum reward per epoch
- **epoch/max_reward**: Maximum reward per epoch
- **epoch/total_samples**: Total samples processed

## Configuration Files

### Hydra Configuration

```yaml
# src/lobster/hydra_config/callbacks/ume_grpo_logging.yaml
_target_: lobster.callbacks.UmeGrpoLoggingCallback

# Logging frequency
log_every_n_steps: 100

# Maximum number of sample examples to log per step
max_samples_to_log: 10

# Whether to log reward histograms
log_reward_histogram: true

# Whether to log individual sample examples
log_sample_examples: true

# Whether to log reward breakdown by modality
log_modality_breakdown: true
```

### Training Configuration

```yaml
# Example training configuration
model_path: "your/base/model"
ume_model_path: "ume-mini-base-12M"
output_dir: "./ume_grpo_runs"
reward_temperature: 0.1
reward_batch_size: 8
enable_wandb_logging: true
wandb_project: "lobster-ume-grpo"
wandb_entity: "your-username"
wandb_run_name: "experiment-001"
```

## Testing

Run the test script to verify logging functionality:

```bash
python examples/test_ume_grpo_logging.py
```

This will test:
- Callback initialization and execution
- Reward function logging
- Trainer creation with logging
- Wandb integration

## Troubleshooting

### Common Issues

1. **Wandb not logging**: Ensure `enable_wandb_logging=True` and wandb is properly configured
2. **No samples logged**: Check that the batch contains completions and rewards
3. **Modality detection fails**: Verify sequence format matches expected patterns

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Considerations

- **Logging frequency**: Reduce `log_every_n_steps` for more frequent updates
- **Sample count**: Limit `max_samples_to_log` to avoid overwhelming wandb
- **Histogram bins**: Adjust histogram resolution as needed

## Example Output

### Wandb Run Configuration

```json
{
  "model_path": "your/base/model",
  "reward_temperature": 0.1,
  "reward_batch_size": 8,
  "output_dir": "./ume_grpo_runs",
  "logging/log_every_n_steps": 100,
  "logging/max_samples_to_log": 10
}
```

### Sample Metrics

```
train/mean_reward: 0.75
train/min_reward: 0.23
train/max_reward: 0.98
train/reward_std: 0.15
train/modality_smiles/mean_reward: 0.82
train/modality_amino_acid/mean_reward: 0.71
train/modality_nucleotide/mean_reward: 0.68
```

### Sample Table

| completion | reward | modality | length |
|------------|--------|----------|--------|
| CC(C)CC1=CC=C(C=C1)C(C)C(=O)O... | 0.82 | smiles | 25 |
| MKTVRQERLKSIVRILERSKEPVSGAQLAE... | 0.71 | amino_acid | 60 |
| ATGCGATCGATCGATCGATCGATCGATCG... | 0.68 | nucleotide | 64 |

## Integration with Existing Code

The enhanced logging is backward compatible. Existing code will work without changes, and logging can be enabled by setting `enable_wandb_logging=True`.

For existing training scripts, simply add the wandb parameters:

```python
# Before
trainer = train_ume_grpo(
    model_path=model_path,
    ume_model_path=ume_model_path,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# After (with logging)
trainer = train_ume_grpo(
    model_path=model_path,
    ume_model_path=ume_model_path,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    enable_wandb_logging=True,
    wandb_project="your-project",
    wandb_entity="your-username",
)
``` 