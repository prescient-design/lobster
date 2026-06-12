#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram/%j.err

# Step W -- velocity + self-cond + Proteina-style distogram aux loss.
#
# FROM-SCRATCH variant: trains the velocity + selfcond backbone AND the
# new distogram head together from initialisation (no resume from the
# selfcond ckpt). The resume-into-distogram path needs an
# `on_load_checkpoint` hook to fill the missing `distogram_head.*` keys
# from init values; until that's wired, this script trains the whole
# model from scratch so the state_dict matches the new arch byte-for-byte.
#
# Distogram knobs (model config defaults):
#   K=64 buckets, max=22 A (AF2 / OpenFold convention)
#   t > 0.5 gate (active in clean-end half of the trajectory)
#   weight=0.25 (won't dominate the velocity loss)
#
# Watch:
#   * train_aux_distogram (should drop within first ~5k steps once the
#     pair head receives gradient)
#   * train_aux_distogram_no_gate (un-gated CE for visibility)
#   * val/loss + val/rmsd_kabsch_n_steps_50 + val/3di_recovery -- compare
#     against `flow_nokabsch_velocity_selfcond` at matched training step.

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
