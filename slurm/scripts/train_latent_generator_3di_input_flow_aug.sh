#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_aug
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_aug/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_aug/%j.err

# Step H companion to the small-plain flow run: same architecture
# (~20.6 M params, 6 layers x 8 heads x 32 dh = 256 inner), same dataset,
# same trainer knobs, but trains with per-step random SO(3) augmentation
# of `x_1` (Proteina-style, train-only). A/Bs against job 12478043
# (small + plain, pinned no-aug) to isolate the augmentation effect.
#
# Fresh run dir / log dir / job name / wandb project so there is zero
# collision with any of the in-flight runs:
#   - /cv/scratch/u/lisanzas/latent_generator_3di_input_flow_aug/runs/
#   - /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_aug/
#   - wandb project `lobster_latent_generator_3di_input_flow_aug`

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_aug/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_aug
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# `random_so3_aug` defaults to true in the model yaml after Step H, so
# no explicit override is needed here. Keep this comment so anyone
# editing the script later knows where the augmentation flag lives.
uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_aug \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
