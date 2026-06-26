#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0/%j.err

# Ablation: weight=0 sibling of `base_3di_coord_ce`.
#
# Resumes from a `flow_nokabsch_velocity_base` checkpoint (no
# coord-CE in its training history) and trains with the coord-CE
# *gradient* disabled (weight=0) while still tracking the diagnostic
# (CE + argmax accuracy logged each step) so we can compare to the
# weight=0.10 champion.
#
# Resume LR:
#   The picked checkpoint (val_loss=0.5042, e=532) was at the lowest
#   val_loss point of the base run. Use the same `LrOverrideOnResume`
#   pattern as the correct_cb launcher to re-warm the LR and keep
#   training stable.
#
# Watch on W&B:
#   train_loss                          -- main flow loss (no aux gradient)
#   train_aux_3di_coord_ce              -- diagnostic only, should track
#                                          the loss curve from the
#                                          weight=0.10 sibling for an
#                                          apples-to-apples comparison.
#   train_aux_3di_coord_acc             -- argmax accuracy (per residue).
#   val/rmsd_kabsch_n_steps_50          -- compare to weight=0.10 run's
#                                          ~9 A on the 30-protein val.

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Best-val_loss checkpoint of THIS w=0 run so far (epoch=603, val=0.3999,
# saved 2026-06-19 22:58 -- 7.5h into the original 5e-5 resume). That
# resume drifted upward on val Kab (~25 -> ~28 by step 54k); restart
# from the in-run best snapshot at 3e-5 to dampen the instability.
# Symlinked to a `=`-free path (Hydra override parser tokenizes on `=`).
CKPT_RESUME='/cv/scratch/u/lisanzas/champion_3di_flow_velocity_w0/w0_e603_s33220_val0p3999.ckpt'

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce_w0 \
    model.ckpt_path="${CKPT_RESUME}" \
    +callbacks.lr_override_on_resume._target_=lobster.callbacks.LrOverrideOnResume \
    +callbacks.lr_override_on_resume.lr=3e-5 \
    +callbacks.lr_override_on_resume.reset_scheduler_step=true \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
