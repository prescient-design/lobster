# Lobster Evaluation Callbacks

This guide explains how to create and use callbacks for model evaluation with the `evaluate_model_with_callbacks` function.

## Available Callbacks

### Evaluation Callbacks

#### PerturbationScoreCallback

Analyzes model robustness through sequence perturbations by measuring cosine distances between original and perturbed embeddings.

**Credits:** Josh Southern for the original perturbation analysis notebook.

**Features:**
- **Shuffling Analysis**: Measures distance between original and randomly shuffled sequences
- **Mutation Analysis**: Measures distance between original and single-point mutation sequences  
- **Visualization**: Generates heatmaps showing mutation sensitivity at each position
- **Metrics**: Computes average distances and their ratio

**Usage:**
```python
from lobster.callbacks import PerturbationScoreCallback
from lobster.constants import Modality

# Example protein sequence
protein_sequence = "QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNT"

callback = PerturbationScoreCallback(
    sequence=protein_sequence,
    modality=Modality.AMINO_ACID,
    num_shuffles=10,
    output_dir="perturbation_analysis",
    save_heatmap=True
)

# Run evaluation
metrics = callback.evaluate(model)
```

**Requirements:**
- Model must implement `embed_sequences(sequences, modality, aggregate)` method
- For training integration, can be added to Lightning Trainer callbacks

#### SklearnProbeCallback (Base Class)

Base class for evaluating embedding models using scikit-learn model probes. Provides infrastructure for training linear probes on embeddings and evaluating their performance.

**Key Features:**
- Supports classification and regression tasks
- Cross-validation and train/test split evaluation
- Dimensionality reduction with PCA
- Multiple probe types (linear, elastic net, SVM)

#### CalmSklearnProbeCallback

Evaluates embedding models on the CALM dataset collection for cDNA sequence property prediction tasks.

**Available Tasks:**
- `meltome`: Protein melting temperature (regression)
- `solubility`: Protein solubility (regression) 
- `localization`: Cellular localization (multilabel, 10 classes)
- `protein_abundance`: Protein abundance (regression, species-specific)
- `transcript_abundance`: Transcript abundance (regression, species-specific)
- `function_bp`: Gene Ontology biological process terms (multilabel, 5 classes)
- `function_cc`: Gene Ontology cellular component terms (multilabel, 5 classes)
- `function_mf`: Gene Ontology molecular function terms (multilabel, 5 classes)

**Usage:**
```python
from lobster.callbacks import CalmSklearnProbeCallback

callback = CalmSklearnProbeCallback(
    tasks=["meltome", "solubility", "localization"],
    species=["hsapiens", "ecoli"],
    batch_size=32,
    probe_type="linear",
    max_samples=3000
)
```

#### PEERSklearnProbeCallback

Evaluates model embeddings on PEER benchmark tasks using sklearn probes. By default evaluates 16 out of 17 PEER tasks (excludes PROTEINNET due to memory issues).

**Usage:**
```python
from lobster.callbacks import PEERSklearnProbeCallback

callback = PEERSklearnProbeCallback(
    batch_size=32,
    probe_type="linear",
    ignore_errors=True
)
```

#### MoleculeACESklearnProbeCallback

Specialized sklearn probe callback for MoleculeACE dataset evaluation.

#### DGEBEvaluationCallback

Evaluates UME and ESM models on DGEB benchmark tasks for biological sequence models.

**Features:**
- Supports both UME models (requires tokenization) and ESM models (raw sequences)
- Protein and DNA modality support
- Configurable batch size, sequence length, and pooling strategies

**Usage:**
```python
from lobster.callbacks import DGEBEvaluationCallback

callback = DGEBEvaluationCallback(
    model_name="UME",
    modality="protein",
    batch_size=32,
    max_seq_length=1024,
    requires_tokenization=True,
    output_dir="dgeb_results"
)
```

### Visualization Callbacks

#### UmapVisualizationCallback

Creates UMAP visualizations of model embeddings with optional grouping by dataset or other attributes.

**Usage:**
```python
from lobster.callbacks import UmapVisualizationCallback

callback = UmapVisualizationCallback(
    output_dir="umap_viz",
    max_samples=1000,
    group_by="dataset",
    n_neighbors=300,
    min_dist=1.0
)
```

### Performance Callbacks

#### TokensPerSecondCallback

Measures model inference speed during training by tracking tokens processed per second across multi-GPU setups.

**Usage:**
```python
from lobster.callbacks import TokensPerSecondCallback

callback = TokensPerSecondCallback(
    log_interval_steps=500,
    verbose=True
)
```

### Utility Callbacks

#### DataLoaderCheckpointCallback

Saves dataloader checkpoints during training for resumability.

#### AuxiliaryTaskWeightScheduler & MultiTaskWeightScheduler

Schedulers for managing auxiliary task loss weights during multi-task training.

#### UmeGrpoLoggingCallback

Specialized logging callback for UME GRPO training.

## Available Imports

All callbacks can be imported from `lobster.callbacks`:

```python
from lobster.callbacks import (
    # Evaluation callbacks
    SklearnProbeCallback,
    SklearnProbeTaskConfig,
    CalmSklearnProbeCallback,
    PEERSklearnProbeCallback,
    MoleculeACESklearnProbeCallback,
    PerturbationScoreCallback,
    DGEBEvaluationCallback,
    
    # Visualization callbacks
    UmapVisualizationCallback,
    
    # Performance callbacks
    TokensPerSecondCallback,
    default_batch_size_fn,
    default_batch_length_fn,
    
    # Utility callbacks
    DataLoaderCheckpointCallback,
    AuxiliaryTaskWeightScheduler,
    MultiTaskWeightScheduler,
    UmeGrpoLoggingCallback,
)
```

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

## Using Callbacks for Evaluation

### Command Line Evaluation

```bash
# Run evaluation using Hydra configuration
lobster_eval model.ckpt_path=path_to_your_checkpoint.ckpt
```

### Programmatic Evaluation

```python
from lobster.evaluation import evaluate_model_with_callbacks
from lobster.callbacks import (
    PerturbationScoreCallback, 
    CalmSklearnProbeCallback,
    UmapVisualizationCallback,
    TokensPerSecondCallback
)
import lightning as L
from torch.utils.data import DataLoader

# Load model from checkpoint
model = L.LightningModule.load_from_checkpoint("path/to/checkpoint.ckpt")

# Create evaluation callbacks
callbacks = [
    PerturbationScoreCallback(
        sequence="QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNT",
        modality="amino_acid",
        num_shuffles=10
    ),
    CalmSklearnProbeCallback(
        tasks=["meltome", "solubility"],
        batch_size=32
    ),
    UmapVisualizationCallback(
        output_dir="umap_viz",
        group_by="dataset"
    )
]

# Optional dataloader for callbacks that need it
dataloader = DataLoader(...)

# Run evaluation
results, report_path = evaluate_model_with_callbacks(
    callbacks=callbacks,
    model=model,
    dataloader=dataloader,
    output_dir="evaluation_results"
)

print(f"Evaluation report saved to: {report_path}")
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