#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_medium/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_medium/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_pl_med_resume
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Ligand Training Script - MEDIUM MODEL RESUME
#
# Resumes training from the medium model sweep checkpoint:
#   - model_size=medium, batch_size=36, lr=1e-3
#   - Checkpoint at step 23666/50000 (epoch 164, val_loss=1.9353)
#   - Effective batch size: 36 * 8 GPUs * 20 accumulate = 5760
#   - SE3 augmentation + ligand minimization enabled
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_protein_ligand_medium_resume.sh

nvidia-smi

# Change to lobster directory (required for hydra config paths)
cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
# Disable EFA network plugin for intra-node - use NVLink/PCIe instead
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Directory setup
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_medium

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Resume from checkpoint
CHECKPOINT="/cv/scratch/u/lisanzas/gen_ume_protein_ligand/runs/2026-02-04T13-33-39/epoch=164-step=23666-val_loss=1.9353.ckpt"

# Medium model with sweep settings: bs=36, lr=1e-3, accumulate=20
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=36 \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=20 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    logger.name=gen_ume_protein_ligand-medium_resume_bs36_lr1e-3 \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true \
    "model.ckpt_path='${CHECKPOINT}'"
