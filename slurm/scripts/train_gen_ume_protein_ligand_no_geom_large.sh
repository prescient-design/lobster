#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_no_geom_large/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_no_geom_large/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_pl_lrg_ng
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Ligand Training Script - LARGE MODEL, NO GEOM
# 
# This trains the LARGE unified model (~740M params) on:
# - PDB: protein-only structures (~50%)
# - PDBBind: protein-ligand complexes (~25%)
# - SAIR: protein-ligand complexes (~25%)
#
# NO ligand-only (GEOM) data - better for protein-centric tasks
# Uses joint protein-ligand encoding for proper interaction learning
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_protein_ligand_no_geom_large.sh

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
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_large/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_no_geom_large

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Large model (~740M params) without GEOM data
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand_no_geom \
    model.encoder_kwargs.model_size=large \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    logger.name=gen_ume_protein_ligand-large_no_geom_joint_encoding \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true

