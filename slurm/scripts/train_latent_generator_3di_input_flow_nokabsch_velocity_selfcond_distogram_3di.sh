#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di/%j.err

# Step X -- velocity + self-cond + distogram aux + 3Di-token CE aux.
#
# From-scratch variant. Both auxiliary heads are zero-init at the final
# layer; the resume-into-aux-heads path is supported via
# `Tokenizer3diInputFlow.on_load_checkpoint` (fills missing keys from
# init), but launching from scratch keeps the comparison vs the
# distogram-only run (16149190) clean -- both runs start with the same
# state, just different aux loss weights.
#
# Knobs (from the model yaml):
#   distogram   K=64 / max=22 A / t>0.5 / weight=0.25
#   3di-CE      no time gate / weight=0.05 (small so the cond-branch
#               trivial inverse can't hijack the backbone gradients)
#
# Watch:
#   * train_aux_distogram + _no_gate                (pair-head CE)
#   * train_aux_3di_ce    + _no_gate + _aux_3di_acc (token-head CE + accuracy)
#   * val/rmsd_kabsch_n_steps_50 / val/3di_recovery vs 16149190 at
#     matched training step.

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distogram_3di \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
