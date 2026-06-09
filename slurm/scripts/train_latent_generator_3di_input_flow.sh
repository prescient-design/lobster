#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow/%j.err

# 3Di-conditioned flow-matching backbone-coord generator (Step E of the
# SwissProt LG plan). Same dataset as the deterministic 3Di-input run
# (`data/structure_pdb_with_3di.yaml`) but trained with
# bionemo-moco's ContinuousFlowMatcher + Kabsch augmentation + Proteina-
# style CFG. Run directories are separate from the deterministic job
# so the two can run side-by-side without filesystem collision.

set -euo pipefail

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# `model.random_so3_aug=false` PIN: Step H made random SO(3) augmentation
# the default in `latent_generator_3di_input_flow.yaml` (and lobster_train
# re-instantiates the model from yaml on every restart). This run started
# BEFORE Step H, so the pin keeps it as a clean no-aug baseline -- the
# matching `train_latent_generator_3di_input_flow_aug.sh` runs the aug
# variant in a separate wandb project / scratch dir.
uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow \
    model.random_so3_aug=false \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
