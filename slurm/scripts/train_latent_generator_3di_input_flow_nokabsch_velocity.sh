#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity/%j.err

# Step S -- plain flow with no Kabsch coupling AND velocity prediction.
# Same as `train_latent_generator_3di_input_flow_nokabsch.sh` but the
# moco interpolant is set to `prediction_type: velocity`. The loss
# `target_type` follows automatically (`_moco_loss` reads it from the
# interpolant), so the loss is `||v_hat - (x_1 - x_0)||^2` with no
# `1/(1-t)^2` reweight (Proteina's `target_pred=v` mode).
#
# Two changes vs. the data-pred sibling -- they are conflated by Proteina
# convention (Step P Tier-1 recommends both for the same low-t-undertrained
# pathology). The natural target_type==prediction_type pairing is the
# default; isolating the parametrization vs. reweight axis is a one-flag
# follow-up if needed.
#
# Fresh run dir / log dir / job name / wandb project:
#   - /cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity/runs/
#   - /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity/
#   - wandb project `lobster_latent_generator_3di_input_flow_nokabsch_velocity`

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
