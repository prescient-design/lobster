#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distill
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distill/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distill/%j.err

# Step V -- velocity_selfcond distillation on PDB + AFDB + synthetic
# (denovo + TED + CATH). Resumes from the Phase 6 family leader
# (`flow_nokabsch_velocity_selfcond` val-best, e=1503/s=82720/val=0.5378,
# 13.99 A Kabsch / 19.0% 3DR at the Phase 3c SDE winner config) and
# trains it on the wider UME structure mix to see if exposure to a
# broader fold distribution closes the residual sampling gap to GT.
#
# Validation moves to AFDB CAMEO (the wider mix's val set), so the
# first val cycle defines the new baseline -- val_loss is NOT
# comparable to the 0.5378 PDB-only number.
#
# We resume the FULL Lightning state (model + Adam moments + scheduler).
# The selfcond run was training cleanly at lr=1e-4 with no divergence,
# so no LrOverrideOnResume callback is wired in; if gradients on the
# new data spike, kill and re-launch with:
#
#     +callbacks.lr_override_on_resume._target_=lobster.callbacks.LrOverrideOnResume \
#     +callbacks.lr_override_on_resume.lr=5e-5 \
#     +callbacks.lr_override_on_resume.reset_scheduler_step=true
#
# Watch:  val/loss (new CAMEO baseline), val/rmsd_kabsch_n_steps_50,
# and gradient norms on the first ~5k steps for any spike.

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distill/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distill
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Symlink to /cv/scratch/.../epoch=1503-step=82720-val_loss=0.5378.ckpt
# under a `=`-free name so Hydra's override parser doesn't choke on the
# embedded `=` characters in the filename.
CKPT_RESUME='/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_selfcond/runs/2026-06-08T19-40-54/resume_e1503_s82720_valloss0.5378.ckpt'

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_distill \
    model.ckpt_path="${CKPT_RESUME}" \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
