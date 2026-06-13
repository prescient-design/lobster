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

# Resume `flow_nokabsch_velocity_base` from the round-3 Kabsch-champion
# snapshot at a further-lowered LR of 5e-5 (was 8e-5 in the prior resume).
#
# Why: the prior resume (8e-5, started from val-best e=532) ran cleanly
# for ~36 h and reached `last.ckpt` Kab=13.2514 A / 3DR=20.94% (eval
# round 3, the family Kab champion captured at
# `/cv/scratch/u/lisanzas/champion_3di_flow_velocity/champion.ckpt`).
# Round 4 (this morning) eval'd a fresher `last.ckpt` and saw Kab spike
# to 17.31 A / 3DR=15.0% -- +4 A regression in ~10 h of additional
# training. Same pattern as the original 1e-4 divergence, just slower:
# 8e-5 is still too aggressive for this ckpt's Adam state. Rolling back
# to the r3 champion ckpt and halving LR to 5e-5.
#
# The champion.ckpt is a copy of the r3 base/last; it has FULL Lightning
# state (optimizer, scheduler, step counter). Lightning's
# `Trainer.fit(ckpt_path=...)` would restore the ckpt's `param_groups[0]['lr']`
# (whatever value it was at when r3 was saved -- post-warmup from the
# 8e-5 schedule, so 8e-5) and clobber any cmd-line `model.optim.lr=...`.
# `LrOverrideOnResume` runs at `on_train_start` AFTER the restore and
# rewrites `param_groups['lr']`, `param_groups['initial_lr']`, and the
# scheduler's `base_lrs` to the new LR. `reset_scheduler_step=true` rewinds
# the constant-warmup so the run gets a soft re-warmup at 5e-5 from 0
# (5000 steps to climb back up; smooths the LR transition).
#
# trainer.global_step / epoch are preserved so the run continues from
# step ~37k in wandb (a fresh wandb run will be created since no explicit
# run_id is set; previous run shows up as ended via slurm-side cancel).
#
# Watch: val/loss and val/rmsd_kabsch_n_steps_50 on the first val cycle.
# If still divergent within a day, drop further (2e-5 or 1e-5).

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

# Round-3 Kabsch champion (saved by phase 7 + the persistent
# `_KabChampion` tracker in the eval scripts). Path is `=`-free so Hydra
# parses it cleanly without the symlink dance the older resume used.
CKPT_RESUME='/cv/scratch/u/lisanzas/champion_3di_flow_velocity/champion.ckpt'

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_base \
    model.ckpt_path="${CKPT_RESUME}" \
    +callbacks.lr_override_on_resume._target_=lobster.callbacks.LrOverrideOnResume \
    +callbacks.lr_override_on_resume.lr=5e-5 \
    +callbacks.lr_override_on_resume.reset_scheduler_step=true \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
