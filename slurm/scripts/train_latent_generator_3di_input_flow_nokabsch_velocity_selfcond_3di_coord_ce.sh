#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_selfcond_3di_coord_ce
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_3di_coord_ce/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_3di_coord_ce/%j.err

# Step Y -- velocity + self-cond + differentiable-mini3di CE-from-coords aux.
#
# Resume from `flow_nokabsch_velocity_selfcond` val-best e1503 so the
# velocity backbone starts already trained; the new aux loss only has
# to incrementally shape the geometry to match Foldseek's 3Di
# assignments at the GT-defined partner pairs. The selfcond ckpt has
# no `mini3di_torch.*` keys but `Tokenizer3diInputFlow.on_load_checkpoint`
# (Step W) tolerates missing keys -- the mini3di buffers are FROZEN
# Foldseek weights so init from the bundled .kerasify file is correct.
#
# Aux config (model yaml):
#   weight=0.10, t_lim=0.5, temperature=1.0
#
# Watch:
#   train_aux_3di_coord_ce / _no_gate
#   train_aux_3di_coord_acc   <- argmax recovery vs GT 3Di tokens
#   val/rmsd_kabsch_n_steps_50, val/3di_recovery vs the selfcond
#   baseline at matched training step.

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_selfcond_3di_coord_ce/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_3di_coord_ce
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# selfcond e1503 val-best (=-free symlink, same as the distill resume).
CKPT_RESUME='/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_selfcond/runs/2026-06-08T19-40-54/resume_e1503_s82720_valloss0.5378.ckpt'

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_selfcond_3di_coord_ce \
    model.ckpt_path="${CKPT_RESUME}" \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
