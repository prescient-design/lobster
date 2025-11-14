# Interactive Protein Inference and Intervention Workflows

This directory contains interactive notebooks and automated scripts for protein dataset analysis, inference, and concept-based interventions using Lobster models.

## 📋 Overview

### New Interactive Notebooks

1. **05-interactive-protein-inference.ipynb**
   - Dataset selection (FASTA files, direct input, or examples)
   - Model selection and configuration (PMLM, CBM, PCLM, UME)
   - Inference with multiple pooling strategies
   - Biological metrics computation
   - Visualization of embeddings and properties
   - Export results to files

2. **06-interactive-protein-intervention.ipynb**
   - Concept-based protein engineering
   - Interactive concept selection from 718 available concepts
   - Configurable intervention parameters
   - Sequence comparison and alignment
   - Biological metrics before/after intervention
   - Concept change analysis
   - Export intervention results

### Automated Scripts

- **scripts/setup_environment.sh** - Environment setup and dependency installation
- **scripts/protein_inference_workflow.py** - Command-line interface for automated workflows

## 🚀 Quick Start

### 1. Environment Setup

Run the setup script to install all dependencies:

```bash
bash scripts/setup_environment.sh
```

This will:
- Sync dependencies using `uv`
- Install notebook dependencies (ipywidgets, jupyter, matplotlib, etc.)
- Install BioPython and Levenshtein for metrics

Then activate the environment:

```bash
source .venv/bin/activate
```

### 2. Run Interactive Notebooks

Start Jupyter:

```bash
jupyter notebook notebooks/
```

Open either:
- `05-interactive-protein-inference.ipynb` for inference workflows
- `06-interactive-protein-intervention.ipynb` for intervention workflows

Follow the step-by-step instructions in each notebook!

### 3. Use Automated CLI Scripts

For batch processing or automation, use the command-line script:

#### Inference Example

```bash
python scripts/protein_inference_workflow.py inference \
    --fasta test_data/query.fasta \
    --model pmlm \
    --checkpoint asalam91/lobster_24M \
    --pooling both \
    --output outputs/inference_results
```

#### Intervention Example

```bash
python scripts/protein_inference_workflow.py intervene \
    --sequence "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF" \
    --concept gravy \
    --direction negative \
    --edits 5 \
    --iterations 3 \
    --output outputs/intervention_results
```

#### List Available Concepts

```bash
python scripts/protein_inference_workflow.py list-concepts \
    --checkpoint asalam91/cb_lobster_24M
```

## 📖 Detailed Usage

### Interactive Inference Notebook

The inference notebook (`05-interactive-protein-inference.ipynb`) provides:

#### Data Sources
- **FASTA Files**: Load from existing FASTA files
- **Direct Input**: Paste sequences directly
- **Examples**: Pre-loaded example sequences (G-protein, peptides, antibodies)

#### Model Types
- **PMLM**: Masked language model (BERT-style) for embeddings
- **CBM-PMLM**: Concept bottleneck model with 718 interpretable concepts
- **PCLM**: Causal language model (GPT-style)
- **UME**: Multimodal embeddings

#### Pooling Strategies
- CLS token: Use [CLS] token representation
- Mean pooling: Average across all tokens
- Max pooling: Maximum across all tokens
- Both: Generate both CLS and mean embeddings

#### Metrics Computed
- Sequence length
- Molecular weight
- Aromaticity
- Instability index
- Isoelectric point
- GRAVY (hydrophobicity)
- Secondary structure fractions (helix, turn, sheet)

#### Visualizations
- PCA projection of embeddings
- Embedding norm distributions
- Concept activation heatmaps (CBM models)
- Biological property distributions
- Property correlations

### Interactive Intervention Notebook

The intervention notebook (`06-interactive-protein-intervention.ipynb`) provides:

#### Sequence Sources
- Direct sequence input
- Example sequences with different properties
- FASTA file loading

#### Concept Selection
Choose from 718 biological concepts including:
- **Biophysical**: molecular_weight, aromaticity, gravy, isoelectric_point
- **Stability**: instability_index, charge_at_pH6, charge_at_pH7
- **Structure**: helix_fraction, turn_structure_fraction, sheet_structure_fraction
- **Spectroscopy**: molar_extinction_coefficient (reduced/oxidized)
- **Surface properties**: avg_hydrophilicity, avg_surface_accessibility
- And 704 more learned concepts!

#### Intervention Configuration
- **Direction**: Positive (increase) or Negative (decrease) concept value
- **Edits per iteration**: Number of amino acid changes (1-20)
- **Iterations**: Number of refinement rounds (1-10)

#### Analysis Features
- Original sequence concept analysis
- Top/bottom concepts visualization
- Iteration-by-iteration tracking
- Edit distance calculation
- Biological metrics comparison
- Sequence alignment display
- Concept change heatmaps

### Automated CLI Workflow

#### Inference Command

```bash
python scripts/protein_inference_workflow.py inference [OPTIONS]
```

Options:
- `--fasta PATH`: Path to FASTA file
- `--sequence SEQ`: Direct sequence input
- `--model {pmlm,cbm,pclm}`: Model type (default: pmlm)
- `--checkpoint ID`: Model checkpoint (default: asalam91/lobster_24M)
- `--pooling {cls,mean,max,both}`: Pooling method (default: both)
- `--output DIR`: Output directory (default: outputs/inference_results)
- `--device {auto,cuda,cpu}`: Device to use (default: auto)

Output files:
- `cls_embeddings.npy`: CLS token embeddings
- `mean_embeddings.npy`: Mean pooled embeddings
- `concepts.npy`: Concept values (CBM models only)
- `sequence_metrics.csv`: Biological metrics
- `sequence_ids.txt`: Sequence identifiers
- `summary.txt`: Run summary

#### Intervention Command

```bash
python scripts/protein_inference_workflow.py intervene [OPTIONS]
```

Options:
- `--fasta PATH`: Path to FASTA file (uses first sequence)
- `--sequence SEQ`: Direct sequence input
- `--concept NAME`: Target concept (default: gravy)
- `--direction {positive,negative}`: Intervention direction (default: negative)
- `--edits N`: Number of edits per iteration (default: 5)
- `--iterations N`: Number of iterations (default: 1)
- `--checkpoint ID`: Model checkpoint (default: asalam91/cb_lobster_24M)
- `--output DIR`: Output directory (default: outputs/intervention_results)
- `--device {auto,cuda,cpu}`: Device to use (default: auto)

Output files:
- `sequences.txt`: Original and modified sequences
- `metrics_comparison.csv`: Biological metrics comparison
- `intervention_trajectory.csv`: Iteration-by-iteration results
- `summary.txt`: Intervention summary

## 🎯 Use Cases

### 1. Protein Property Analysis

Analyze a set of proteins to understand their properties:

```bash
python scripts/protein_inference_workflow.py inference \
    --fasta my_proteins.fasta \
    --model cbm \
    --output analysis/proteins
```

### 2. Reduce Protein Hydrophobicity

Make a protein more hydrophilic:

```bash
python scripts/protein_inference_workflow.py intervene \
    --fasta protein.fasta \
    --concept gravy \
    --direction negative \
    --edits 10 \
    --iterations 5 \
    --output analysis/hydrophilic_variant
```

### 3. Increase Protein Stability

Target stability-related concepts:

```bash
python scripts/protein_inference_workflow.py intervene \
    --sequence "YOURPROTEINSEQUENCE" \
    --concept instability_index \
    --direction negative \
    --edits 5 \
    --iterations 3 \
    --output analysis/stable_variant
```

### 4. Explore Concept Space

Use the interactive notebooks to:
1. Load your protein
2. View all 718 concept values
3. Identify concepts to optimize
4. Run interventions interactively
5. Compare before/after properties
6. Export results for further analysis

## 📊 Understanding Results

### Embeddings

Embeddings are high-dimensional vector representations of proteins:
- **CLS embeddings**: Single vector per sequence from [CLS] token
- **Mean embeddings**: Average of all token embeddings
- Use for: Similarity search, clustering, downstream ML tasks

### Concepts (CBM Models)

Concepts are interpretable biological features:
- **Values**: Normalized scores (typically 0-1 range)
- **High values**: Protein has more of that property
- **Low values**: Protein has less of that property

### Interventions

Interventions modify sequences to change concept values:
- **Edit distance**: Number of amino acid changes
- **Concept change**: Δ in target concept value
- **Sequence identity**: % of unchanged positions
- **Side effects**: Changes in other concepts

## 🔧 Troubleshooting

### CUDA Out of Memory

Reduce batch size or use CPU:
```bash
--device cpu
```

### Missing Dependencies

Re-run environment setup:
```bash
bash scripts/setup_environment.sh
```

### Concept Not Found

List all available concepts:
```bash
python scripts/protein_inference_workflow.py list-concepts
```

### Jupyter Widgets Not Displaying

Install and enable widgets:
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## 📚 Additional Resources

- [Lobster Documentation](../README.md)
- [Original Inference Notebook](01-inference.ipynb)
- [Original Intervention Notebook](02-intervention.ipynb)
- [Model Checkpoints](https://huggingface.co/asalam91)

## 🤝 Contributing

Found a bug or have a feature request? Please open an issue on the GitHub repository.

## 📝 Citation

If you use these workflows in your research, please cite the Lobster paper and repository.

---

**Happy protein engineering! 🧬**
