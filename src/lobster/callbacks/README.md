# Lobster Evaluation Callbacks

This guide explains how to create and use callbacks for model evaluation with the `evaluate_model_with_callbacks` function.

## Creating an Evaluation Callback

Create a new callback by subclassing `lightning.Callback` and implementing an `evaluate` method:

```python
import lightning as L
from pathlib import Path
from typing import Any, dict

class MyEvaluationCallback(L.Callback):
    def __init__(self, param1: str | None = None):
        super().__init__()
        self.param1 = param1
        
    def evaluate(self, model: L.LightningModule) -> dict[str, Any]:
        """Evaluate the model and return metrics.
        
        Parameters
        ----------
        model : L.LightningModule
            The model to evaluate
            
        Returns
        -------
        dict[str, Any]
            Dictionary of metrics
        """
        # Implement your evaluation logic
        return {"metric1": 0.95, "metric2": 0.85}
```

## Callback Requirements

1. Must inherit from `lightning.Callback`
2. Must implement an `evaluate` method that takes a model as first parameter
3. Can optionally accept a `dataloader` parameter if needed

## Return Types

Your `evaluate` method can return:

- **Dictionary of metrics**: Will be formatted as a table in the report
- **Path to file**: Images will be embedded in the report
- **Nested dictionaries**: Will be formatted as nested tables
- **Other values**: Will be included as text

## Using Your Callback

```python
from lobster.evaluation import evaluate_model_with_callbacks
import lightning as L
from torch.utils.data import DataLoader

# Create model and callbacks
model = MyModel()
callbacks = [
    MyEvaluationCallback(),
    AnotherCallback(param="value")
]

# Optional dataloader for callbacks that need it
dataloader = DataLoader(...)

# Run evaluation
report_path = evaluate_model_with_callbacks(
    callbacks=callbacks,
    model=model,
    dataloader=dataloader,
    output_dir="results/"
)
```

## Advanced: Dataloader-Dependent Callbacks

If your callback needs a dataloader:

```python
def evaluate(self, model: L.LightningModule, dataloader: DataLoader) -> dict[str, float]:
    """Run evaluation that requires a dataloader.
    
    Parameters
    ----------
    model : L.LightningModule
        The model to evaluate
    dataloader : DataLoader
        The dataloader to use for evaluation
        
    Returns
    -------
    dict[str, float]
        Dictionary of metrics
    """
    results = {}
    for batch in dataloader:
        # Process batch with model
        # Update results
    return results
```

The evaluation framework will automatically detect the dataloader parameter and provide it. 