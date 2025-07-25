# Selective Token Modeling

This directory contains experiments related to calculating per-token losses on an existing pretrained model for the purpose of Selective Token Modeling (SLM) (see the Rho-1 [paper](https://arxiv.org/abs/2404.07965) by Lin et al.).
The core idea is that not all tokens are similarly difficult for the model to learn; in the English language, this might be tokens such as `the`. Faster convergence, better performance, and/or reduced model parameter size can be achieved by selectively trains on useful tokens that aligned with the desired distribution. We can make use of previously trained models to determine this notion of "in-distribution".

From the project root directory, running
```
LOBSTER_PROJECT_DIR=$(pwd)
sbatch slurm/scripts/save_token_losses.sh
```

will launch a multi-GPU inference job that saves per-token losses for a FASTA sequence on a specified model (the autoregressive [RITA-Large](https://arxiv.org/abs/2205.05789) model is used by default) into Parquet format.

Model training with selective token percentages can be done using the dataloader in `datasets/_sharded_parquet_dataset.py`.

## Extensions:
- [ ] Perform the same experiment for other modalities for data mixture determination
- [ ] Perform the same experiment for downstream tasks to determine which tasks are more difficult for the model
- [ ] Perform ablation experiments by incorporating data at different loss percentages.
- [ ] Perform on masked language models to see if the pattern is different. Note: this will require O(L) forward passes.
