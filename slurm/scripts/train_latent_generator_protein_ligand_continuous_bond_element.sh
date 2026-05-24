#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_lg_continuous_5dim/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_lg_continuous_5dim/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=lg_cont_5dim
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# =============================================================================
# Protein-Ligand Latent Generator - CONTINUOUS 5-DIM (No Quantization)
# =============================================================================
# This model uses 5-dimensional continuous embeddings (no FSQ/SLQ quantization) with:
#   - Bond matrix embedding: Encodes ligand topology (which atoms are bonded)
#   - Element embedding: Encodes atom types using extended vocabulary (25 tokens)
#
# Key difference from FSQ variant:
#   - model.quantizer=null: No quantization, pure continuous embeddings
#   - embed_dim=5: Ultra-compact 5-dimensional latent space (vs 256-dim)
#
# New features enabled:
#   - use_ligand_bond_embedding=true: Adds BondMatrixEmbedding module
#   - ligand_atom_embedding=true: Adds element type embeddings
#   - use_extended_element_vocab=true: Uses 25-token vocabulary (matches Gen-UME)
#
# Usage:
#   sbatch slurm/scripts/train_latent_generator_protein_ligand_continuous_bond_element.sh
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
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_continuous_5dim/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_lg_continuous_5dim

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Train CONTINUOUS 5-DIM model (no quantization) with bond matrix and element embeddings
# NOTE: For continuous models, embed_dim MUST match struc_token_codebook_size
#       because encoder output goes directly to decoder (no quantizer in between)
#       Using 5-dim latent (similar to MAR's 16-dim per-token diffusion)
uv run lobster_train \
    experiment=train_latent_generator \
    data=structure_ligand_pdb_sair_bond \
    model=latent_generator_ligand_fsq \
    data.num_workers=8 \
    trainer.devices=8 \
    model.num_warmup_steps=10000 \
    model.num_training_steps=100000 \
    model.lr_scheduler.num_warmup_steps=10000 \
    model.lr_scheduler.num_training_steps=100000 \
    model.optim.lr=1e-4 \
    model.quantizer=null \
    'model.structure_encoder.embed_dim=5' \
    'model.structure_encoder.embed_dim_hidden=16' \
    model.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=5 \
    model.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=5 \
    model.structure_encoder.ligand_atom_embedding=true \
    +model.structure_encoder.use_ligand_bond_embedding=true \
    +model.structure_encoder.use_extended_element_vocab=true \
    +callbacks.backbone_reconstruction.use_extended_element_vocab=true \
    trainer.num_sanity_val_steps=2 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    logger.name=lg_continuous_5dim_pdb_sair

