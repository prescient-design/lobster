#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch/%j.err

# Step S -- plain flow with no Kabsch optimal-transport coupling, but
# `random_so3_aug=true` kept on (was previously absorbed by the Kabsch
# alignment; now actually marginalises the loss over orientations).
# Same small (S, ~20.6 M params, 6 x 8 x 32 = 256 inner) architecture,
# same dataset, same trainer knobs as the existing plain-flow run.
# A/Bs against job 12693025 (small + plain + Kabsch + aug) to isolate
# the OT-coupling effect.
#
# Fresh run dir / log dir / job name / wandb project so there is zero
# collision with any of the in-flight runs:
#   - /cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch/runs/
#   - /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch/
#   - wandb project `lobster_latent_generator_3di_input_flow_nokabsch`

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Both `augmentation_type: null` (drop Kabsch) and `random_so3_aug: true`
# (keep SO(3) aug) are set in the model yaml; no explicit overrides needed.
uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
