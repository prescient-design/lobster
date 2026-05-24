#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_pl_diffusion/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_pl_diffusion/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_diffusion
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# =============================================================================
# Gen-UME Protein-Ligand with Diffusion Loss for Structure Tokens
# =============================================================================
#
# This trains Gen-UME using DiffusionLoss (from MAR paper) for structure tokens:
# - Sequence tokens: discrete flow matching (unchanged)
# - Ligand atom tokens: discrete flow matching (unchanged)
# - Bond matrix: categorical CE (unchanged)
# - Structure tokens: DiffusionLoss (continuous embeddings)
#
# Key feature: Instead of predicting discrete structure tokens via CE loss,
# the model predicts continuous structure embeddings using a small diffusion MLP.
# This eliminates the need for vector quantization of structure tokens.
#
# Reference: "Autoregressive Image Generation without Vector Quantization"
# https://arxiv.org/abs/2406.11838
#
# NOTE: This script requires a continuous LatentGenerator checkpoint where
# quantizer=null. The diffusion_target_dim must match the encoder's embed_dim.
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_protein_ligand_diffusion.sh
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
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_pl_diffusion/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_pl_diffusion

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Train Gen-UME with DiffusionLoss for structure tokens
# 
# Key diffusion hyperparameters (from MAR paper defaults):
#   diffusion_depth: 3 (MLP depth)
#   diffusion_width: 1024 (MLP hidden dim)
#   diffusion_num_sampling_steps: "100" (generation steps)
#   diffusion_noise_schedule: cosine (noise schedule)
#   diffusion_loss_weight: 1.0 (relative weight vs other losses)
#
# To tune hyperparameters, add overrides like:
#   model.diffusion_depth=6 \
#   model.diffusion_width=2048 \

uv run lobster_train \
    experiment=train_gen_ume_protein_ligand_diffusion \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2

