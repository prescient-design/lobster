# LOBSTER
**L**anguage m**O**dels for **B**iological **S**equence **T**ransformation and **E**volutionary **R**epresentation

## A language model library for rapid pre-training from scratch.
`lobster` is a language model library for proteins and other biological sequences. Led by [Nathan Frey](https://github.com/ncfrey), [Taylor Joren](github.com/taylormjs), Aya Ismail, and [Allen Goodman](https://github.com/0x00b1), with many valuable contributions from [Contributors](docs/contributors.md) across [Prescient Design](https://www.gene.com/scientists/our-scientists/prescient-design).

This repository contains code and access to pre-trained language models for biological sequence data.

<p align="center">
<img src="assets/lobster.jpeg" width=200px>
</p>

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
- all models are trained on heavy and light chains separately, no pairing
- public antibodies: antiberty dataset (paired and unpaired OAS)
- private antibodies: all tenx (paired) and bulk (unpaired) ngs data
- aho: these models expect aho aligned inputs from anarci
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

## License
Copyright © 2024 Genentech, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.