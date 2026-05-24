#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_large/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_large/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_pl_lrg
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Ligand Training Script - LARGE MODEL (~740M params)
# 
# This trains the unified protein-ligand model on:
# - PDB: protein-only structures (backbone)
# - GEOM: ligand conformers (with bond_matrix)
# - PDBBind: protein-ligand complexes (with bond_matrix)
# - SAIR: protein-ligand complexes (with bond_matrix)
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_protein_ligand_large.sh

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
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_large/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_large

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Large model: ~740M params (hidden_size=1024, num_layers=58)
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    model.encoder_kwargs.model_size=large \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    logger.name=gen_ume_protein_ligand-large_pdb_geom_pbind_sair_bond

