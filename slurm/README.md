# Running LBSTER Jobs with SLURM

This guide explains how to run a `lobster` training job using SLURM on a GPU-enabled system. It also describes which environment variables need to be exported for the job to run properly.

# SLURM Job Script
The provided example job script `scripts/train_ume.sh` is configured up for training the `UME` model on a GPU-enabled SLURM cluster. 

You will need to set specific environment variables to run the job. These will be read by the `UME` hydra configuration file, which is located at `src/lobster/hydra_config/experiment/train_ume.yaml`.

Variables:

* `LOBSTER_DATA_DIR`: Path to the directory containing your training data. Datasets will be downloaded and cached to this directory (if `data.download` is set to `True` in the hydra configuration file).
* `LOBSTER_RUNS_DIR`: Path to the directory where training results (model checkpoints, logs, etc.) will be stored.
* `LOBSTER_USER`: The user entity for the logger (usually your wandb username).
* `WANDB_BASE_URL`: The base URL for the Weights & Biases service. Optional - only needed if you wandb account is not on the default wandb server.

Example:
```bash
    export LOBSTER_DATA_DIR="/data/lobster/ume/data"
    export LOBSTER_RUNS_DIR="/data/lobster/ume/runs"
    export LOBSTER_USER=$(whoami)
    export WANDB_BASE_URL=https://your_org.wandb.io/
```


