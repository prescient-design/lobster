#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce/%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce/%j.err

# Step Y2 -- BASE velocity + differentiable-mini3di CE-from-coords aux loss.
#
# Resume from the round-3 family Kab champion (`base/last`, captured at
# `/cv/scratch/u/lisanzas/champion_3di_flow_velocity/champion.ckpt`,
# Kab=13.2514 A / 3DR=20.94% on the 30-protein PDB val). The champion
# is the base model (768d/12L/12h), so we use the BASE-size yaml here;
# resuming this ckpt into the small selfcond yaml would fail strict
# state_dict load.
#
# LR override to 5e-5 for the same reason as the parallel `base_resume`
# job (16212625): the 8e-5 schedule diverged in round 4. Drop further
# to 5e-5 here and re-warmup over 5000 steps.
#
# Aux loss (model yaml): weight=0.10, t_lim=0.5, temperature=1.0.
# `Tokenizer3diInputFlow.on_load_checkpoint` fills missing
# `mini3di_torch.*` keys from init values (frozen Foldseek weights,
# loaded from the bundled `.kerasify` file -- there's no other source
# we'd want to restore from).
#
# Watch:
#   train_aux_3di_coord_ce / _no_gate
#   train_aux_3di_coord_acc   <- argmax recovery vs GT 3Di tokens
#   val/rmsd_kabsch_n_steps_50, val/3di_recovery vs the parallel
#   `base_resume` job (which has the same LR but no aux loss).

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce
mkdir -p "${LOBSTER_RUNS_DIR}"
mkdir -p "${LOBSTER_DATA_DIR}"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Round-3 Kab champion. Path is `=`-free so Hydra parses it cleanly.
CKPT_RESUME='/cv/scratch/u/lisanzas/champion_3di_flow_velocity/champion.ckpt'

uv run lobster_train \
    experiment=train_latent_generator_3di_input_flow_nokabsch_velocity_base_3di_coord_ce \
    model.ckpt_path="${CKPT_RESUME}" \
    +callbacks.lr_override_on_resume._target_=lobster.callbacks.LrOverrideOnResume \
    +callbacks.lr_override_on_resume.lr=5e-5 \
    +callbacks.lr_override_on_resume.reset_scheduler_step=true \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=0
