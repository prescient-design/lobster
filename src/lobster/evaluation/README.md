# Lobster Model Evaluation

This module provides tools for evaluating models using specialized callbacks and generating comprehensive evaluation reports, including integration with the DGEB benchmark for biological sequence models.

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
results, report_path = evaluate_model_with_callbacks(
    callbacks=callbacks, 
    model=model, 
    dataloader=dataloader, 
    output_dir="evaluation_results"
)

# The report is saved to report_path
print(f"Evaluation report available at: {report_path}")
# You can also inspect the results programmatically
print(results)
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

## DGEB Evaluation

This module includes integration with [DGEB (DNA/Protein Language Model Benchmark)](https://github.com/TattaBio/DGEB) for comprehensive evaluation of UME models on biological sequence tasks.

### Quick Start with DGEB

#### Command Line Interface

The easiest way to run DGEB evaluations is through the command line:

```bash
# Evaluate a pretrained UME model on all protein tasks
uv run lobster_dgeb_eval ume-mini-base-12M --modality protein

# Evaluate a custom checkpoint on DNA tasks
uv run lobster_dgeb_eval /path/to/checkpoint.ckpt --modality dna --max-seq-length 8192

# Run specific tasks only with custom parameters
uv run lobster_dgeb_eval ume-mini-base-12M \
    --modality protein \
    --tasks ec_classification convergent_enzymes_classification \
    --batch-size 64 \
    --output-dir my_dgeb_results
```

#### Python API

You can also run evaluations programmatically:

```python
from lobster.evaluation import UMEAdapterDGEB, run_evaluation

# Run DGEB evaluation
results = run_evaluation(
    model_name="ume-mini-base-12M",
    modality="protein",
    output_dir="dgeb_results",
    batch_size=32,
    max_seq_length=1024,  # DGEB standard - required for benchmark compatibility
)

print(f"Evaluated {len(results['results'])} tasks")
print(f"Results saved to: dgeb_results/")

### Output Files and Reports

DGEB evaluations generate multiple output files in the specified directory:

#### Directory Structure
```
dgeb_results/                          # Output directory (customizable)
├── results_summary.json              # Complete results in JSON format
├── evaluation_report.md              # Human-readable markdown report
└── <model_name>/                     # DGEB's native output format
    └── default/
        ├── ec_classification.json
        ├── convergent_enzymes_classification.json
        └── ... (one file per task)
```

#### Key Output Files

**`evaluation_report.md`** - Human-readable markdown report containing:
- Model configuration and metadata
- Results summary table with primary metrics
- Detailed task-by-task breakdown with all metrics
- Performance notes and context about the evaluation

**`results_summary.json`** - Machine-readable JSON summary with:
- Model metadata (parameters, embedding dimension, etc.)
- Structured results for each task with all metrics
- Evaluation timestamps and configuration
- Easy to parse for further analysis or plotting

**`<model_name>/default/*.json`** - Raw DGEB output files:
- One JSON file per task with complete results
- Includes task metadata, model info, and detailed metrics
- Compatible with DGEB's standard output format

### Available Tasks

DGEB includes 22 tasks across protein and DNA modalities:

#### Protein Tasks (14 total)
- **Classification**: `ec_classification`, `convergent_enzymes_classification`, `MIBIG_protein_classification`
- **Phylogeny**: `rpob_bac_phylogeny`, `rpob_arch_phylogeny`, `fefe_phylogeny`, `bac_16S_phylogeny`, `arch_16S_phylogeny`, `euk_18S_phylogeny`
- **Retrieval**: `arch_retrieval`, `euk_retrieval`
- **Pair Prediction**: `ecoli_operonic_pair`
- **Clustering**: `mopb_clustering`
- **Bigene**: `bacarch_bigene`

#### DNA Tasks (8 total)
- **Classification**: `ec_dna_classification`, `MIBIG_dna_classification`
- **Phylogeny**: `rpob_bac_dna_phylogeny`, `rpob_arch_dna_phylogeny`
- **Pair Prediction**: `cyano_operonic_pair`, `vibrio_operonic_pair`
- **Clustering**: `ecoli_rna_clustering`
- **Bigene**: `modac_paralogy_bigene`

### Model Requirements

#### Pretrained Models
If you have AWS credentials configured, you can use pretrained UME models:
```bash
uv run lobster_dgeb_eval ume-mini-base-12M --modality protein
uv run lobster_dgeb_eval ume-medium-base-480M --modality protein
uv run lobster_dgeb_eval ume-large-base-740M --modality protein
```

#### Custom Models
Any UME model checkpoint can be evaluated:
```bash
uv run lobster_dgeb_eval /path/to/your/checkpoint.ckpt --modality protein
```

### Configuration Options

```bash
uv run lobster_dgeb_eval MODEL_NAME \
    --modality {protein,dna}           # Required: sequence modality
    --tasks TASK1 TASK2 ...            # Optional: specific tasks (default: all)
    --output-dir OUTPUT_DIR            # Optional: results directory (default: dgeb_results)
    --batch-size BATCH_SIZE            # Optional: encoding batch size (default: 32)
    --max-seq-length MAX_LENGTH        # Optional: max sequence length (default: 1024)
    --use-flash-attn                   # Optional: enable flash attention
    --l2-norm                          # Optional: L2-normalize embeddings
    --pool-type {mean,max,cls,last}    # Optional: pooling strategy (default: mean)
    --devices 0 1 2                    # Optional: GPU devices (default: [0])
    --seed 42                          # Optional: random seed (default: 42)
```

**Important Note on Sequence Length**: DGEB expects a maximum sequence length of 1024 tokens, which is our default value. This requirement is consistent with the [DGEB benchmark specification](https://github.com/TattaBio/DGEB/blob/1b187e607e278a34ba7338b8b43747c57add4134/scripts/eval_all_models.py#L10). Changing this value may affect benchmark compatibility and results comparability.

**DNA Tasks**: For DNA tasks, it is recommended to set `--max-seq-length` to 8,192 for optimal performance, as DNA sequences can be significantly longer than protein sequences.

### Example Evaluation Session

```bash
# Run evaluation
uv run lobster_dgeb_eval ume-mini-base-12M --modality protein --output-dir my_eval

# Check results
ls my_eval/
# evaluation_report.md  results_summary.json  ume-mini-base-12M/

# View human-readable report
cat my_eval/evaluation_report.md

# Parse JSON results programmatically
python -c "
import json
with open('my_eval/results_summary.json') as f:
    results = json.load(f)
    print(f'Model: {results[\"model_name\"]}')
    print(f'Tasks: {len(results[\"results\"])}')
    for task in results['results']:
        print(f'  {task[\"task_name\"]}: {task[\"scores\"]}')
"
```

### Performance Tips

- **Batch Size**: Increase `--batch-size` for faster evaluation on GPU (try 64-128)
- **Sequence Length**: The default of 1024 is required for DGEB benchmark compatibility. Only reduce `--max-seq-length` if memory is limited and you understand the impact on benchmark comparability
- **DNA Tasks**: Use `--max-seq-length 8192` for DNA tasks to handle longer sequences effectively
- **Flash Attention**: Use `--use-flash-attn` on GPU for better performance
- **Multiple GPUs**: Use `--devices 0 1 2 3` for multi-GPU inference

### Troubleshooting

**AWS Credentials Error**: Either configure AWS credentials or use a local checkpoint file
**CUDA Out of Memory**: Reduce `--batch-size` or `--max-seq-length`
**Empty Results**: Check logs for sequence filtering warnings

See the [DGEB integration source code](dgeb_adapter.py) for implementation details.

## Advanced: Adding Custom Evaluation Logic

To extend the evaluation framework, create callbacks with custom evaluation logic in the `lobster.callbacks` module. See the [callbacks README](../callbacks/README.md) for more details on creating compatible callbacks. 