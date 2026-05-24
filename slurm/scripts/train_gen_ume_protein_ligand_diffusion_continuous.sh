#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_pl_diffusion_cont/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_pl_diffusion_cont/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_diff_cont
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# =============================================================================
# Gen-UME Protein-Ligand with Diffusion Loss + Continuous LatentGenerator
# =============================================================================
#
# This trains Gen-UME using:
# - DiffusionLoss for structure tokens (from MAR paper)
# - Continuous LatentGenerator (quantizer=null, embed_dim=256)
#
# Key components:
# - Sequence tokens: discrete flow matching (unchanged)
# - Ligand atom tokens: discrete flow matching (unchanged)
# - Bond matrix: categorical CE (unchanged)
# - Structure tokens: DiffusionLoss (continuous 256-dim embeddings)
#
# LatentGenerator model: "LG Protein Ligand cont" (registered in inference.py)
#
# Key settings from continuous LatentGenerator:
#   - quantizer=null (continuous embeddings)
#   - embed_dim=256 (structure embedding dimension)
#   - struc_token_codebook_size=256 (decoder input dim)
#   - use_ligand_bond_embedding=true
#   - use_extended_element_vocab=true
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_protein_ligand_diffusion_continuous.sh
#
# =============================================================================

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
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_pl_diffusion_continuous/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_pl_diffusion_cont

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Train Gen-UME with DiffusionLoss for structure tokens
# Using continuous LatentGenerator (quantizer=null, embed_dim=256)
#
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand_diffusion \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    model.diffusion_target_dim=256 \
    model.diffusion_depth=3 \
    model.diffusion_width=1024 \
    model.use_diffusion_loss_structure=true \
    callbacks.protein_ligand_forward_folding.num_samples=30 \
    logger.name=gen_ume_pl_diffusion_continuous

