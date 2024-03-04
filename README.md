# LOBSTER 🦞
**L**anguage m**O**dels for **B**iological **S**equence **T**ransformation and **E**volutionary **R**epresentation

## A language model library for rapid pre-training from scratch.
`lobster` is a "batteries included" language model library for proteins and other biological sequences. Led by [Nathan Frey](https://github.com/ncfrey), [Taylor Joren](github.com/taylormjs), Aya Ismail, and [Allen Goodman](https://github.com/0x00b1), with many valuable contributions from [Contributors](docs/CONTRIBUTORS.md) across [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design).

This repository contains code and access to pre-trained language models for biological sequence data.

<p align="center">
<img src="assets/lobster.jpeg" width=200px>
</p>

## Notice: Alpha Release
This is an alpha release. The API is subject to change and the documentation is incomplete.

## Why you should use LOBSTER
* LOBSTER is built for pre-training models quickly from scratch. It is "batteries included" This is most useful if you need to control the pre-training data mixture and embedding space, or want to experiment with novel pre-training objectives and fine-tuning strategies. 
* LOBSTER is a living, open-source library that will be periodically updated with new pre-trained models from the [Frey Lab](https://github.com/ncfrey) at [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design). The Frey Lab works on real therapeutic molecule design problems and LOBSTER models and capabilities reflect the demands of real-world drug discovery campaigns.
* LOBSTER is built on [yeji](https://github.com/0x00b1/yeji/tree/main), a standard library for biological research, and integrated with [cortex](https://github.com/prescient-design/cortex/tree/main), a modular framework for multitask modeling, guided generation, and multi-modal models.

## Citations
If you use the code and/or models, please cite:
TODO: update

## Install
clone the repo, cd into it and do `mamba env create -f env.yml`

### Install with pip
from the root of the repo, do
```bash
pip install -e .
```

## Main models you should use
### Loading a pre-trained model
```python
from lobster.model import PrescientPMLM, PrescientPCLM
masked_language_model = PrescientPMLM.load_from_checkpoint(<path to ckpt>)
causal_language_model = PrescientPCLM.load_from_checkpoint(<path to ckpt>)
```
3D and dynamic models use the same classes.

TODO: update

**Datasets**
- all antibody models are trained on heavy and light chains separately, no pairing
- antibodies: antiberty dataset (paired and unpaired OAS)
- aho: these models expect aho aligned antibody inputs from anarci
- uniref50: general protein universe
- pdb complexes: pdb sequences with `.` tokens indicating complexes

**Models**
* PrescientPMLM: masked language model (BERT-style encoder-only architecture)
* PrescientPCLM: causal language model (Llama-style decoder-only architecture)
* PrescientPLMFold: structure prediction language models (pre-trained encoder + structure head)

## Usage

### Embedding
The entrypoint `lobster_embed` is the main driver for embedding sequences and accepts parameters using Hydra syntax. The available parameters for configuration can be found by running `lobster_embed --help` or by looking in the src/lobster/hydra_config directory

To embed a fasta file of sequences using a pre-trained model on an interactive GPU node, cd into the root dir of this repo and do
```bash
lobster_embed data.path_to_fasta="test_data/query.fasta" checkpoint="path_to_checkpoint.ckpt"
```

This will generate a dataframe of embeddings and also [log them to wandb](https://genentech.wandb.io/freyn6/lobster-embedding/runs/luv4ebtv?workspace=user-freyn6).

### Regression and classification
For robust multitask modeling, we recommend using `lobster` with [cortex]((https://github.com/prescient-design/cortex/tree/main)). For simple baselines using `lobster` embeddings, use `lobster.model.LinearProbe`.

### Likelihoods
Likelihoods from an autoregressive `PrescientCLM` or pseudo-log likelihoods ("naturalness") from a `PrescientPMLM` can be computed for a list of `sequences` using

```python
model.naturalness(sequences)
model.likelihood(sequences)
```

## Example Jupyter notebooks

### Protein structure prediction

see [this notebook](notebooks/01-lobster-fold.ipynb) for an example on using PPLMFold to predict structure from sequence.

### Structure-aware sequence embedding with 3D-PPLM
see [this notebook](notebooks/02-3d-lobster.ipynb) for an example on using the [FoldseekTransform](src/lobster/transforms/_foldseek_transforms.py) and 3D-PPLM to embed a monomer or complex.

## Training from scratch
The entrypoint `lobster_train` is the main driver for training and accepts parameters using Hydra syntax. The available parameters for configuration can be found by running `lobster_train --help` or by looking in the src/lobster/hydra_config directory

To train an MLM on a fasta file of sequences on an interactive GPU node, cd into the root dir of this repo and do
```bash
lobster_train data.path_to_fasta="test_data/query.fasta" logger=csv paths.root_dir="."
```

## Contributing
Contributions are welcome! We ask that all users and contributors remember that the LOBSTER team are all full-time drug hunters, and our open-source efforts are a labor of love because we care deeply about open science and scientific progress.

### Install dev requirements and pre-commit hooks

```bash
python -m pip install -r requirements-dev.in
pre-commit install
```

### Testing

```bash
python -m pytest -v --cov-report term-missing --cov=./lobster ./tests
```

### Build and browse docs locally

```bash
make -C docs html
cd docs/build/html
python -m http.server
```

Then open `http://localhost:8000` in your browser.