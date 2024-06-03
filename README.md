# LBSTER 🦞
**L**anguage models for **B**iological **S**equence **T**ransformation and **E**volutionary **R**epresentation

## A language model library for rapid pre-training from scratch.
`lobster` is a "batteries included" language model library for proteins and other biological sequences. Led by [Nathan Frey](https://github.com/ncfrey), [Taylor Joren](github.com/taylormjs), [Aya Ismail](https://github.com/ayaabdelsalam91), and [Allen Goodman](https://github.com/0x00b1), with many valuable contributions from [Contributors](docs/CONTRIBUTORS.md) across [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design).

This repository contains code and access to pre-trained language models for biological sequence data.

<!---
image credit: Amy Wang
-->
<p align="center">
<img src="assets/lobster.png" width=200px>
</p>

## Notice: Alpha Release
This is an alpha release. The API is subject to change and the documentation is incomplete.
*LBSTER is a work-in-progress. Contributions and feedback are encouraged!*

<details open><summary><b>Table of contents</b></summary>

- [Why you should use LBSTER](#why-use)
- [Citations](#citations)
- [Install instructions](#install)
- [Models](#main-models)
- [Usage](#usage)
</details>

## Why you should use LBSTER <a name="why-use"></a>
* LBSTER is built for pre-training models quickly from scratch. It is "batteries included." This is most useful if you need to control the pre-training data mixture and embedding space, or want to experiment with novel pre-training objectives and fine-tuning strategies.
* LBSTER is a living, open-source library that will be periodically updated with new code and pre-trained models from the [Frey Lab](https://ncfrey.github.io/) at [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design). The Frey Lab works on real therapeutic molecule design problems and LBSTER models and capabilities reflect the demands of real-world drug discovery campaigns.
* LBSTER is built with [beignet](https://github.com/Genentech/beignet/tree/main), a standard library for biological research, and integrated with [cortex](https://github.com/prescient-design/cortex/tree/main), a modular framework for multitask modeling, guided generation, and multi-modal models.

## Citations <a name="citations"></a>
If you use the code and/or models, please cite the relevant papers.
For the `lbster` code base:
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


## Install <a name="install"></a>
clone the repo, cd into it and do `mamba env create -f env.yml`
then from the root of the repo, do
```bash
pip install -e .
```

## Main models you should use <a name="main-models"></a>
### Loading a pre-trained model
```python
from lobster.model import LobsterPMLM, LobsterPCLM
masked_language_model = LobsterPMLM.load_from_checkpoint(<path to ckpt>)
causal_language_model = LobsterPCLM.load_from_checkpoint(<path to ckpt>)
```
3D, cDNA, and dynamic models use the same classes.

NOTE: Pre-trained model checkpoints *may* be included in future releases!

**Models**
* LobsterPMLM: masked language model (BERT-style encoder-only architecture)
* LobsterPCLM: causal language model (Llama-style decoder-only architecture)
* LobsterPLMFold: structure prediction language models (pre-trained encoder + structure head)

## Usage <a name="usage"></a>

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
Likelihoods from an autoregressive `PrescientCLM` or pseudo-log likelihoods ("naturalness") from a `PrescientPMLM` can be computed for a list of `sequences` using

```python
model.naturalness(sequences)
model.likelihood(sequences)
```

## Training from scratch
The entrypoint `lobster_train` is the main driver for training and accepts parameters using Hydra syntax. The available parameters for configuration can be found by running `lobster_train --help` or by looking in the src/lobster/hydra_config directory

To train an MLM on a fasta file of sequences on an interactive GPU node, cd into the root dir of this repo and do
```bash
lobster_train data.path_to_fasta="test_data/query.fasta" logger=csv paths.root_dir="."
```

## Contributing
Contributions are welcome! We ask that all users and contributors remember that the LBSTER team are all full-time drug hunters, and our open-source efforts are a labor of love because we care deeply about open science and scientific progress.

### Install dev requirements and pre-commit hooks

```bash
python -m pip install -r requirements-dev.in
pre-commit install
```

### Testing

```bash
python -m pytest -v --cov-report term-missing --cov=./lobster ./tests
```
