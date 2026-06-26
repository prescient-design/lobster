#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_correct_cb
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_correct_cb/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_correct_cb/%j.err

# Step Y3 -- BASE velocity + mini3di CE-from-coords aux, resumed from
# the prior 3Di-CE champion to re-fit the input embedding against the
# corrected 3Di tokens.
#
# Resume policy:
#   The prior run's input-embedding table was learned against a different
#   3Di token distribution; re-warm the LR via
#   `LrOverrideOnResume(lr=5e-5, reset_scheduler_step=true)` so the
#   embedding can re-fit.
#
# Watch on W&B:
#   train_loss        -- expect a spike at step 0, then a 1-3 epoch
#                         transition while the embedding adapts.
#   train_aux_3di_coord_ce / _acc  -- new reference distribution.
#   val/rmsd_kabsch_n_steps_50  -- compare to the prior champion's
#                                   13.11 A. Should drop substantially
#                                   as the model leverages the corrected
#                                   3Di conditioning.

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_correct_cb/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_correct_cb
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Day-4 champion (refreshed 2026-06-19 20:32, val Kab=6.55, AAR=76.5%
# on the 30-protein PDB val with WINNER_W2). Original 5e-5 resume
# diverged after ~5 days (val Kab climbed from ~10 to ~38 by step 115k);
# restart from the day-4 champion at 3e-5 (~1.7x lower) to dampen the
# instability while keeping the embedding-refit dynamic the run was
# designed around.
CKPT_RESUME='/cv/scratch/u/lisanzas/champion_3di_flow_velocity_correct_cb_w2/champion.ckpt'

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_correct_cb \
    model.ckpt_path="${CKPT_RESUME}" \
    +callbacks.lr_override_on_resume._target_=lobster.callbacks.LrOverrideOnResume \
    +callbacks.lr_override_on_resume.lr=3e-5 \
    +callbacks.lr_override_on_resume.reset_scheduler_step=true \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
