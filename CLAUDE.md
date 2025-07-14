# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

LBSTER (Lobster) is a "batteries included" language model library for proteins and biological sequences built with PyTorch Lightning and Hydra configuration management. The project uses a modular architecture with clear separation between data handling, model definitions, training, and evaluation.

### Key Directories

- `src/lobster/` - Main package containing all core functionality
- `src/lobster/model/` - Model architectures (MLM, CLM, concept bottleneck models, etc.)
- `src/lobster/data/` - Data modules for various biological datasets
- `src/lobster/datasets/` - Dataset implementations for different data types
- `src/lobster/tokenization/` - Tokenizers for biological sequences (amino acids, nucleotides, SMILES)
- `src/lobster/transforms/` - Data transformation functions
- `src/lobster/evaluation/` - Model evaluation tools and benchmarks (DGEB integration, callbacks)
- `src/lobster/hydra_config/` - Hydra configuration files for all components
- `src/lobster/cmdline/` - Command-line interface implementations
- `tests/` - Unit tests mirroring the src structure
- `examples/` - Example scripts for inference and interventions
- `notebooks/` - Jupyter notebooks for tutorials

## Common Commands

### Installation and Environment
```bash
# Primary installation method with uv
uv sync
uv sync --extra flash  # With flash attention

# Alternative with mamba
mamba env create -f env.yml
pip install -e .
```

### Testing
```bash
python -m pytest -v --cov-report term-missing --cov=./lobster ./tests
```

### Code Quality
```bash
pre-commit install  # Setup pre-commit hooks
pre-commit run --all-files  # Run all checks
ruff check --fix  # Lint and fix issues
ruff format  # Format code
```

### Training and Inference
```bash
# Train a model
lobster_train data.path_to_fasta="test_data/query.fasta" logger=csv paths.root_dir="."

# Embed sequences
lobster_embed data.path_to_fasta="test_data/query.fasta" checkpoint="path_to_checkpoint.ckpt"

# Other CLI commands
lobster_predict
lobster_intervene
lobster_perplexity
lobster_eval
lobster_dgeb_eval ume-mini-base-12M --modality protein
```

## Model Architecture

The library implements several model types:

1. **LobsterPMLM**: Masked Language Model (BERT-style encoder-only)
2. **LobsterCBMPMLM**: Concept Bottleneck Masked Language Model with 718 biological concepts
3. **LobsterPCLM**: Causal Language Model (Llama-style decoder-only)
4. **LobsterPLMFold**: Structure prediction models (encoder + structure head)
5. **Modern BERT variants**: Including FlexBERT and Hyena architectures

### Key Base Classes
- `LMBaseForMaskedLM` - Base class for masked language models
- `LMBaseContactPredictionHead` - Contact prediction head for structure tasks
- Models extend PyTorch Lightning modules for training management

## Configuration System

The project uses Hydra for configuration management:

- All configs in `src/lobster/hydra_config/`
- Modular configs: `data/`, `model/`, `callbacks/`, `trainer/`, etc.
- Default training config uses `fasta` data and `mlm` model
- Override configs with command-line syntax: `model=clm data=calm`

## Tokenization

Multiple specialized tokenizers for biological sequences:
- `pmlm_tokenizer` - Default protein tokenizer
- `amino_acid_tokenizer` - Amino acid sequences
- `nucleotide_tokenizer` - DNA/RNA sequences  
- `smiles_tokenizer` - Chemical SMILES strings
- `hyena_tokenizer`, `mgm_tokenizer` - Architecture-specific tokenizers

All tokenizers follow HuggingFace tokenizer interface and include vocabulary files and special tokens.

## Data Pipeline

Data handling follows Lightning's DataModule pattern:
- `FastaDataModule` - FASTA sequence files
- `CALMDataModule` - CALM dataset
- `UMEDataModule` - UME multimodal data
- `ChemblDataModule` - Chemical property data

Datasets implement PyTorch's Dataset/IterableDataset interface with support for distributed training and efficient data loading.

## Development Notes

- Project uses `uv` for dependency management (preferred) or `mamba`
- Pre-commit hooks enforce ruff linting and formatting
- Test coverage expected for new code
- Models support both CPU and GPU execution
- Distributed training supported via Lightning
- Integration with Weights & Biases for experiment tracking

## Code Standards

### Type Hints
- Use modern Python 3.10+ union syntax: `str | None` instead of `Optional[str]`
- Use built-in generics: `list[str]`, `dict[str, Any]` instead of `List[str]`, `Dict[str, Any]`
- All functions must have comprehensive type hints for parameters and return values
- Minimize `typing` imports - prefer built-in types and union operator `|`

### Documentation
- Follow NumPy docstring conventions for all public functions and classes
- Include all standard sections: Parameters, Returns, Raises, Examples, Notes, See Also
- Provide realistic examples relevant to biological sequence modeling
- Document all parameters with types and clear descriptions
- Specify all possible exceptions in Raises section

## Git Commit Guidelines

When creating git commits, DO NOT include the following text in commit messages:
```
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

Keep commit messages concise and focused on the actual changes made.

## Updating This File

Keep CLAUDE.md updated when making changes to:
- CLI commands or entry points
- Project structure or key directories
- Installation methods or dependencies
- Core model architectures
- Configuration patterns or data pipeline