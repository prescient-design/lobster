# LBSTER 🦞
**L**anguage models for **B**iological **S**equence **T**ransformation and **E**volutionary **R**epresentation


`lobster` is a "batteries included" language model library for proteins and other biological sequences. Led by [Nathan Frey](https://github.com/ncfrey), [Taylor Joren](https://github.com/taylormjs), [Aya Abdlesalam Ismail](https://github.com/ayaabdelsalam91), [Joseph Kleinhenz](https://github.com/kleinhenz) and [Allen Goodman](https://github.com/0x00b1), with many valuable contributions from [Contributors](docs/CONTRIBUTORS.md) across [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design).

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
- [Training and inference](#training)
- [Contributing](#contributing)
</details>

## Why you should use LBSTER <a name="why-use"></a>
* LBSTER is built for pre-training models quickly from scratch. It is "batteries included." This is most useful if you need to control the pre-training data mixture and embedding space, or want to experiment with novel pre-training objectives and fine-tuning strategies.
* LBSTER is a living, open-source library that will be periodically updated with new code and pre-trained models from the [Frey Lab](https://ncfrey.github.io/) at [Prescient Design, Genentech](https://www.gene.com/scientists/our-scientists/prescient-design). The Frey Lab works on real therapeutic molecule design problems and LBSTER models and capabilities reflect the demands of real-world drug discovery campaigns.
* LBSTER is built with [beignet](https://github.com/Genentech/beignet/tree/main), a standard library for biological research, and integrated with [cortex](https://github.com/prescient-design/cortex/tree/main), a modular framework for multitask modeling, guided generation, and multi-modal models.
* LBSTER supports concepts; we have a concept-bottleneck protein language model we refer to as CB-LBSTER, which supports 718 concepts.

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
Lobster_150M | 150 M | uniref50 | 24M parameter protein Masked LLM trained on uniref50|[lobster_150M](https://huggingface.co/asalam91/lobster_150M)


#### CB LMs
| Shorthand | #params | Dataset |  Description  | Model checkpoint |
|---------|------------|---------|------------------------------------------------------------|-------------|
cb_Lobster_24M | 24 M | uniref50+SwissProt | 24M parameter a protein concept bottleneck model for protiens with 718 concepts | [cb_lobster_24M](https://huggingface.co/asalam91/cb_lobster_24M)
cb_Lobster_150M | 150 M | uniref50+SwissProt |150M parameter a protein  concept bottleneck model for protiens with 718 concepts|[cb_lobster_150M](https://huggingface.co/asalam91/cb_lobster_150M)
cb_Lobster_650M | 650 M | uniref50+SwissProt |650M parameter  a protein concept bottleneck model for protiens with 718 concepts|[cb_lobster_650M](https://huggingface.co/asalam91/cb_lobster_650M)
cb_Lobster_3B | 3 B | uniref50+SwissProt |3B parameter  a protein concept bottleneck model for protiens with 718 concepts|[cb_lobster_3B](https://huggingface.co/asalam91/cb_lobster_3B)

### Loading a pre-trained model
```python
from lobster.model import LobsterPMLM, LobsterPCLM, LobsterCBMPMLM
masked_language_model = LobsterPMLM("asalam91/lobster_mlm_24M")
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

Check out [jupyter notebook tutorial](notebooks/01-inference.ipynb) for example on how extract embedding reprsentations from different models.


### Concept Interventions

Check out [jupyter notebook tutorial](notebooks/02-intervention.ipynb) for example on to intervene on different concepts for our concept-bottleneck models class.


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
Likelihoods from an autoregressive `PrescientCLM` or pseudo-log likelihoods ("naturalness") from a `PrescientPMLM` can be computed for a list of `sequences` using

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

## Contributing <a name="contributing"></a>
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
