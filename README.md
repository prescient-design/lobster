# LBSTER 🦞
**L**anguage models for **B**iological **S**equence **T**ransformation and **E**volutionary **R**epresentation


`lobster` is a "batteries included" language model library for proteins and other biological sequences. Led by [Nathan Frey](https://github.com/ncfrey), [Karina Zadorozhny](https://github.com/karinazad), [Taylor Joren](https://github.com/taylormjs), [Sidney Lisanza](https://github.com/Sidney-Lisanza), [Aya Abdlesalam Ismail](https://github.com/ayaabdelsalam91), [Joseph Kleinhenz](https://github.com/kleinhenz) and [Allen Goodman](https://github.com/0x00b1), with many valuable contributions from [Contributors](docs/CONTRIBUTORS.md) across [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design).

This repository contains training code and access to pre-trained language models for biological sequence data.

## Usage


<!---
image credit: Amy Wang
-->
<p align="center">
<img src="https://raw.githubusercontent.com/prescient-design/lobster/refs/heads/main/assets/lobster.png" width=200px>
</p>


<details open><summary><b>Table of contents</b></summary>

- [Why you should use LBSTER](#why-use)
- [Citations](#citations)
- [Install instructions](#install)
- [Models](#main-models)
- [Notebooks](#notebooks)
- [MCP Server](#mcp-integration)
- [Training and inference](#training)
- [Reinforcement Learning with UME](#rl-training)
- [Contributing](#contributing)
</details>

## Why you should use LBSTER <a name="why-use"></a>
* LBSTER is built for pre-training models quickly from scratch. It is "batteries included." This is most useful if you need to control the pre-training data mixture and embedding space, or want to experiment with novel pre-training objectives and fine-tuning strategies.
* LBSTER is a living, open-source library that will be periodically updated with new code and pre-trained models from the [Frey Lab](https://ncfrey.github.io/) at [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design). The Frey Lab works on real therapeutic molecule design problems and LBSTER models and capabilities reflect the demands of real-world drug discovery campaigns.
* LBSTER is built with [beignet](https://github.com/Genentech/beignet/tree/main), a standard library for biological research, and integrated with [cortex](https://github.com/prescient-design/cortex/tree/main), a modular framework for multitask modeling, guided generation, and multi-modal models.
* LBSTER supports concepts; we have a concept-bottleneck protein language model, CB-LBSTER, which supports 718 concepts.

## Citations <a name="citations"></a>
If you use the code and/or models, please cite the relevant papers.
For the `lbster` code base cite: [Cramming Protein Language Model Training in 24 GPU Hours](https://www.biorxiv.org/content/early/2024/05/15/2024.05.14.594108)
```bibtex
@article{Frey2024.05.14.594108,
	author = {Frey, Nathan C. and Joren, Taylor and Ismail, Aya Abdelsalam and Goodman, Allen and Bonneau, Richard and Cho, Kyunghyun and Gligorijevi{\'c}, Vladimir},
	title = {Cramming Protein Language Model Training in 24 GPU Hours},
	elocation-id = {2024.05.14.594108},
	year = {2024},
	doi = {10.1101/2024.05.14.594108},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/05/15/2024.05.14.594108},
	eprint = {https://www.biorxiv.org/content/early/2024/05/15/2024.05.14.594108.full.pdf},
	journal = {bioRxiv}
}

```


For the `cb-lbster` code base cite: [Concept Bottleneck Language Models for Protein Design](https://arxiv.org/abs/2411.06090)
```bibtex
@article{ismail2024conceptbottlenecklanguagemodels,
      title={Concept Bottleneck Language Models For protein design}, 
      author={Aya Abdelsalam Ismail and Tuomas Oikarinen and Amy Wang and Julius Adebayo and Samuel Stanton and Taylor Joren and Joseph Kleinhenz and Allen Goodman and Héctor Corrada Bravo and Kyunghyun Cho and Nathan C. Frey},
      year={2024},
      eprint={2411.06090},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.06090}, 
}

```

## Install <a name="install"></a>

### Using `uv`
Install [uv](https://github.com/astral-sh/uv) and create a new virtual environment:


```bash
uv venv --python 3.12  # create a new virtual environment in the `lobster` directory
source .venv/bin/activate
uv pip install -e .
```

Alternatively, run installation directly with `uv sync`:
```bash
uv sync
uv sync --all-extras --no-cache  # to resolve flash-attn installation issues
```

and then prefix every command with `uv run`. For example,

```bash
uv run lobster_train data.path_to_fasta="test_data/query.fasta" 
```

### flash attention
To make use of [flash attention](https://github.com/Dao-AILab/flash-attention) install with the `flash` extra
```bash
uv sync --extra flash
```

### Using `mamba`
clone the repo, cd into it and do `mamba env create -f env.yml`
then from the root of the repo, do
```bash
pip install -e .
```

## Main models you should use <a name="main-models"></a>

### Pretrained Models

#### Masked LMs
| Shorthand | #params | Dataset |  Description  | Model checkpoint |
|---------|------------|---------|------------------------------------------------------------|-------------|
Lobster_24M | 24 M | uniref50 | 24M parameter protein Masked LLM trained on uniref50| [lobster_24M](https://huggingface.co/asalam91/lobster_24M)
Lobster_150M | 150 M | uniref50 | 150M parameter protein Masked LLM trained on uniref50|[lobster_150M](https://huggingface.co/asalam91/lobster_150M)


#### CB LMs
| Shorthand | #params | Dataset |  Description  | Model checkpoint |
|---------|------------|---------|------------------------------------------------------------|-------------|
cb_Lobster_24M | 24 M | uniref50+SwissProt | 24M parameter a protein concept bottleneck model for proteins with 718 concepts | [cb_lobster_24M](https://huggingface.co/asalam91/cb_lobster_24M)
cb_Lobster_150M | 150 M | uniref50+SwissProt |150M parameter a protein  concept bottleneck model for proteins with 718 concepts|[cb_lobster_150M](https://huggingface.co/asalam91/cb_lobster_150M)
cb_Lobster_650M | 650 M | uniref50+SwissProt |650M parameter  a protein concept bottleneck model for proteins with 718 concepts|[cb_lobster_650M](https://huggingface.co/asalam91/cb_lobster_650M)
cb_Lobster_3B | 3 B | uniref50+SwissProt |3B parameter  a protein concept bottleneck model for proteins with 718 concepts|[cb_lobster_3B](https://huggingface.co/asalam91/cb_lobster_3B)

### Loading a pre-trained model
```python
from lobster.model import LobsterPMLM, LobsterPCLM, LobsterCBMPMLM
masked_language_model = LobsterPMLM("asalam91/lobster_24M")
concept_bottleneck_masked_language_model = LobsterCBMPMLM("asalam91/cb_lobster_24M")
causal_language_model = LobsterPCLM.load_from_checkpoint(<path to ckpt>)
```
3D, cDNA, and dynamic models use the same classes.

**Models**
* LobsterPMLM: masked language model (BERT-style encoder-only architecture)
* LobsterCBMPMLM: concept bottleneck masked language model (BERT-style encoder-only architecture with a concept bottleneck and a linear decoder)
* LobsterPCLM: causal language model (Llama-style decoder-only architecture)
* LobsterPLMFold: structure prediction language models (pre-trained encoder + structure head)


## Notebooks <a name="notebooks"></a>

### Representation learning

Check out this [jupyter notebook tutorial](notebooks/01-inference.ipynb) for an example on how to extract embedding reprsentations from different models.


### Concept Interventions

Check out this [jupyter notebook tutorial](notebooks/02-intervention.ipynb) for an example on how to intervene on different concepts for our concept-bottleneck models class.

## MCP Integration

Lobster supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) for seamless integration with Claude Desktop and other AI tools:

```bash
# Install with MCP support
uv sync --extra mcp

# Setup Claude Desktop integration
uv run lobster_mcp_setup
```

<!-- ### One-click install for Cursor -->
<!-- TODO -->
<!-- [![Add Lobster MCP server to Cursor](https://cursor.com/deeplink/mcp-install-dark.svg)](cursor://anysphere.cursor-deeplink/mcp/install?name=lobster-inference&config=eyJjb21tYW5kIjoidXYiLCJhcmdzIjpbInJ1biIsIi0tZXh0cmEiLCJtY3AiLCJsb2JzdGVyX21jcF9zZXJ2ZXIiXX0=) -->

After setup, you can use Lobster models directly in Cursor or Claude Desktop with natural language commands like:
- "Get embeddings for this protein sequence using lobster_24M"
- "What concepts are supported by the cb_lobster_24M model?"
- "Intervene on this sequence to reduce hydrophobicity"

See [MCP Integration Guide](docs/MCP_INTEGRATION.md) for complete documentation.

## DXT Extension for Claude Desktop

Lobster is available as a **DXT (Desktop Extension Toolkit) extension** for Claude Desktop, providing a one-click installation experience:

### Quick Install

1. **Download**: Get the latest `.dxt` file from [GitHub Releases](https://github.com/prescient-design/lobster/releases)
2. **Install**: Double-click the `.dxt` file or drag it into Claude Desktop
3. **Use**: Start using Lobster models with natural language commands

### Features

- **One-click installation** - No command line setup required
- **Self-contained** - Includes all dependencies (~500MB)
- **Automatic updates** - New versions available through GitHub Releases
- **Full functionality** - All MCP server capabilities included

### Usage Examples

Once installed, you can use natural language commands in Claude Desktop:

```
What Lobster models are available for protein analysis?

Get embeddings for the sequence MKTVRQERLKSIVRIL using lobster_24M

What concepts are supported by the cb_lobster_24M model?

Intervene on MKTVRQERLKSIVRIL to reduce hydrophobicity using cb_lobster_24M
```

### Development

For developers who want to build and test DXT extensions locally:

```bash
# Build DXT extension locally
python scripts/build_dxt.py

# Create a release (updates version, builds, and creates GitHub release)
python scripts/release_dxt.py 0.1.0
```

See [DXT Distribution Guide](docs/DXT_DISTRIBUTION.md) for detailed build and distribution instructions.

## Example scripts

Check out [examples](examples/) for scripts showing how to perform inference and interventions.

## Training and inference <a name="training"></a>

### Embedding
The entrypoint `lobster_embed` is the main driver for embedding sequences and accepts parameters using Hydra syntax. The available parameters for configuration can be found by running `lobster_embed --help` or by looking in the src/lobster/hydra_config directory

To embed a fasta file of sequences using a pre-trained model on an interactive GPU node, cd into the root dir of this repo and do
```bash
lobster_embed data.path_to_fasta="test_data/query.fasta" checkpoint="path_to_checkpoint.ckpt"
```

This will generate a dataframe of embeddings and also log them to wandb.

### Regression and classification
For robust multitask modeling, we recommend using `lobster` with [cortex]((https://github.com/prescient-design/cortex/tree/main)). For simple baselines using `lobster` embeddings, use `lobster.model.LinearProbe` and `lobster.model.LobsterMLP`.

### Likelihoods
Likelihoods from an autoregressive `LobsterCLM` or pseudo-log likelihoods ("naturalness") from a `LobsterPMLM` can be computed for a list of `sequences` using

```python
model.naturalness(sequences)
model.likelihood(sequences)
```

### Training from scratch
The entrypoint `lobster_train` is the main driver for training and accepts parameters using Hydra syntax. The available parameters for configuration can be found by running `lobster_train --help` or by looking in the src/lobster/hydra_config directory

To train an MLM on a fasta file of sequences on an interactive GPU node, cd into the root dir of this repo and do
```bash
lobster_train data.path_to_fasta="test_data/query.fasta" logger=csv paths.root_dir="."
```

### Reinforcement Learning with UME Reward Functions <a name="rl-training"></a>

Lobster supports reinforcement learning training using UME-based reward functions for post-training language models. This approach uses UME pseudo-likelihood scores as rewards to guide model behavior toward generating more biologically plausible sequences.

**Quick Start:**
```bash
# Step 1: Generate synthetic dataset
cd examples
python generate_synthetic_dataset.py

# Step 2: Run UME-based GRPO training
python train_ume_grpo.py
```

**Key Features:**
- **Automatic modality detection** for SMILES, amino acid, and DNA sequences
- **UME-based reward functions** using pseudo-likelihood scores
- **GRPO training** with TRL integration
- **Modular design** with reusable components

For detailed instructions and advanced usage, see the [RL Training Guide](docs/RL_TRAINING.md).

## Contributing <a name="contributing"></a>
Contributions are welcome! We ask that all users and contributors remember that the LBSTER team are all full-time drug hunters, and our open-source efforts are a labor of love because we care deeply about open science and scientific progress.

### Getting started with contributions
Expanding unit test coverage, docstrings, and type hints are always welcome and a good place to start to orient yourself to the code base. Likewise for identifying and fixing 🐛bugs🐛. For more involved project ideas, check [Good First Issues](https://github.com/prescient-design/lobster/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22). All new or modified code *must* be unit tested before maintainers will review.

### Install dev requirements and pre-commit hooks

```bash
pre-commit install
```

### Create lockfile for env
```bash
uv pip compile requirements.in -o requirements.txt
```

### Testing

```bash
python -m pytest -v --cov-report term-missing --cov=./lobster ./tests
```
