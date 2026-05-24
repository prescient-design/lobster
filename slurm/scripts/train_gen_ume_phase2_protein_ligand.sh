#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_twophase/%J_phase2.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_twophase/%J_phase2.err
#SBATCH --job-name=genume-ph2-pl
#SBATCH -t 7-00:00:00

# =============================================================================
# Two-Phase Training — Phase 2: Switch to Protein-Ligand Data
#
# Resumes from Phase 1 checkpoint (protein-only pretrained) and continues
# training on the full protein-ligand dataset (PDB + AFDB + PDBBind + SAIR +
# PLINDER + Distillation + Redesign).
#
# Smaller batch size (28) due to ligand overhead.
# Effective batch: 28 * 8 GPUs * 25 accumulate = 5,600
#
# Set PHASE1_CKPT to the best Phase 1 checkpoint before submitting:
#   PHASE1_CKPT=/cv/scratch/u/lisanzas/gen_ume_twophase/runs/<run>/last.ckpt \
#     sbatch slurm/scripts/train_gen_ume_phase2_protein_ligand.sh
# =============================================================================

set -euo pipefail

PHASE1_CKPT="${PHASE1_CKPT:?Set PHASE1_CKPT=/path/to/phase1/last.ckpt}"

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "Phase 1 checkpoint: ${PHASE1_CKPT}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_twophase/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_twophase
mkdir -p /cv/scratch/u/lisanzas/gen_ume_twophase/runs

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    data=structure_ligand_all \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=28 \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=25 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    trainer.precision=bf16-mixed \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true \
    logger.name=gen_ume_twophase-phase2_protein_ligand_medium \
    logger.project=lobster_gen_ume_protein_ligand \
    "model.ckpt_path='${PHASE1_CKPT}'"
