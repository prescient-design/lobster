#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_base
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base/%j.err

# Step U -- velocity-prediction at BASE size (no Kabsch, no self-cond).
# ~110.6 M params: 768d / 12L / 12h * 64dh = 768 inner attn dim,
# time_cond_dim=256. Same recipe as `flow_nokabsch_velocity` at the
# BASE scale. LR kept at 1e-4 to A/B cleanly against the small velocity
# run; if the BASE model diverges (val_loss unstable, NaN, gradient
# explosion), edit this script to add `+model.optim.lr=8e-5` (or
# smaller -- the pre-Step-Q muP scaling for the 1.5x width jump) and
# re-launch. Watch `val/loss` and `val/rmsd_kabsch_n_steps_50` on the
# first val cycle.

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_base/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_base \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
