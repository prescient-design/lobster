# Lobster Model Evaluation

This module provides tools for evaluating models using specialized callbacks and generating comprehensive evaluation reports.

## How Evaluation Works

The `evaluate_model_with_callbacks` function runs each callback's `evaluate` method on a model, collects the results, and generates a markdown report. The process follows these steps:

1. Initialize output directory
2. For each callback:
   - Call its `evaluate` method with the model (and dataloader if required)
   - Collect results or log any issues
3. Generate a markdown report with all results
4. Return the path to the report

## Using the Evaluation Framework

To run evaluation in the command line:

```bash
lobster_eval model.model_path=path_to_your_checkpoint.ckpt
```

To run evaluation in code:
```python
from lobster.evaluation import evaluate_model_with_callbacks
from lobster.callbacks import LinearProbeCallback, UMAPVisualizationCallback
import lightning as L
import torch

# Prepare model and evaluation dataset
model = L.LightningModule.load_from_checkpoint("path/to/checkpoint.ckpt")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Define callbacks for different evaluation tasks
callbacks = [
    LinearProbeCallback(num_classes=10),
    UMAPVisualizationCallback(n_components=2),
]

# Run evaluation
report_path = evaluate_model_with_callbacks(
    callbacks=callbacks, 
    model=model, 
    dataloader=dataloader, 
    output_dir="evaluation_results"
)

# The report is saved to report_path
print(f"Evaluation report available at: {report_path}")
```

## Evaluation Report Format

The generated report is a markdown file with the following structure:

- Title and evaluation date
- Evaluation results section with subsections for each callback
- Each callback's results are formatted based on their type:
  - Dictionaries are displayed as tables
  - Nested dictionaries create nested tables
  - Paths to files (like images) are embedded
  - Other results are included as plain text
- A section listing any issues encountered during evaluation

## Example Report

```markdown
# Model Evaluation Report

Evaluation date: 2023-08-15 14:30:22

## Evaluation Results

### LinearProbeCallback

| Metric | Value |
|--------|-------|
| accuracy | 0.8750 |
| f1_score | 0.8523 |
| precision | 0.8912 |
| recall | 0.8157 |

### UMAPVisualizationCallback

![UMAPVisualizationCallback Visualization](/path/to/umap_plot.png)

### TokensPerSecondCallback

| Metric | Value |
|--------|-------|
| tokens_per_second | 1256.3400 |
| batch_size | 32 |

### PeerEvaluationCallback

### ResNet50

| Metric | Value |
|--------|-------|
| perplexity | *see below* |

**Nested metrics for perplexity:**

| Submetric | Value |
|-----------|-------|
| avg | 4.2100 |
| min | 1.0500 |
| max | 12.3600 |

## Issues Encountered

- CalmLinearProbeCallback: No suitable data found for probing task
```

## Advanced: Adding Custom Evaluation Logic

To extend the evaluation framework, create callbacks with custom evaluation logic in the `lobster.callbacks` module. See the [callbacks README](../callbacks/README.md) for more details on creating compatible callbacks. 