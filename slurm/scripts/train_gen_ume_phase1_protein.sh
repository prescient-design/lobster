#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_twophase/%J_phase1.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_twophase/%J_phase1.err
#SBATCH --job-name=genume-ph1-prot
#SBATCH -t 7-00:00:00

# =============================================================================
# Two-Phase Training — Phase 1: Protein-Ligand Model on Protein-Only Data
#
# Uses the ProteinLigandEncoderLightningModule but trains on protein-only data
# (PDB + AFDB + Denovo + TED + CATH, SS-balanced). Larger batch size (48)
# since there's no ligand overhead.
#
# Effective batch: 48 * 8 GPUs * 20 accumulate = 7,680
#
# After Phase 1 completes, switch to Phase 2 with protein-ligand data:
#   slurm/scripts/train_gen_ume_phase2_protein_ligand.sh
# =============================================================================

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
    data=structure_pdb_afdb_denovo_ted_cath_ss_balanced \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=48 \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=20 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    trainer.precision=bf16-mixed \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    logger.name=gen_ume_twophase-phase1_protein_only_medium \
    logger.project=lobster_gen_ume_protein_ligand
