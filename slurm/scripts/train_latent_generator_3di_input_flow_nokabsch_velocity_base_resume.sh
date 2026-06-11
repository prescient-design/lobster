#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_base_resume
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_resume/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_resume/%j.err

# Resume `flow_nokabsch_velocity_base` from its val-best snapshot
# (epoch=532-step=29315, val_loss=0.5042) at a lowered LR of 8e-5.
#
# Why: the original 1e-4 launch (Step U) ran for ~600 epochs and started
# climbing past its val minimum. Phase 5b sampling on the live ckpt
# returned 20.49 A Kabsch / 14.8% 3DR, vs 14.22 A / 18.9% on the
# val-best snapshot -- a 6.3 A regression. We're rolling back and
# lowering the LR to the slurm-script-comment-recommended 8e-5
# (matches the muP scaling for the 1.5x width jump from the small
# velocity run that's training cleanly at 1e-4).
#
# Lightning's `Trainer.fit(ckpt_path=...)` restores the optimizer state
# from the ckpt, which overwrites `param_groups[0]['lr']` with the
# original 1e-4 -- so passing `model.optim.lr=8e-5` on the command line
# alone gets clobbered. We use the `LrOverrideOnResume` callback (added
# below via Hydra) which fires at `on_train_start` (after the ckpt
# restore is complete) and rewrites `param_groups['lr']`,
# `param_groups['initial_lr']`, and the scheduler's `base_lrs` to 8e-5.
# `reset_scheduler_step=true` also rewinds the constant-warmup scheduler
# so the run gets a soft re-warmup over 5000 steps from 0 -> 8e-5.
#
# trainer.global_step / epoch are preserved so the run continues from
# step 29315 in wandb (a fresh wandb run will be created since no
# explicit run_id is set).
#
# Watch:  val/loss and val/rmsd_kabsch_n_steps_50 on the first val cycle.
# If still divergent within a day, drop further (5e-5 or 4e-5).

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

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_resume
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Symlink to the original epoch=532-step=29315-val_loss=0.5042.ckpt
# under a `=`-free name so Hydra's override parser doesn't choke on the
# embedded `=` characters.
CKPT_RESUME='/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_base/runs/2026-06-08T19-25-48/resume_e532_s29315_valloss0.5042.ckpt'

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_base \
    model.ckpt_path="${CKPT_RESUME}" \
    +callbacks.lr_override_on_resume._target_=lobster.callbacks.LrOverrideOnResume \
    +callbacks.lr_override_on_resume.lr=8e-5 \
    +callbacks.lr_override_on_resume.reset_scheduler_step=true \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
