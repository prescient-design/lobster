#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_lg_fsq_sair_240_4375/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_lg_fsq_sair_240_4375/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=lg_fsq_sair_240_4375
#SBATCH -t 7-00:00:00
#SBATCH -q llm

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
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/latent_generator_fsq_sair_240_4375/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_lg_fsq_sair_240_4375

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# FSQ with SAIR dataset (12_15_25 with bond_matrix)
# FSQ levels: [7,5,5,5,5] = 4375 tokens for both ligand
# FSQ levels: [8,6,5,1,1] = 240 tokens for protein
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
    'model.quantizer.protein_levels=[5,4,3,2,2]' \
    'model.quantizer.ligand_levels=[7,5,5,5,5]' \
    model.decoder_factory.decoder_mapping.vit_decoder.struc_token_codebook_size=240 \
    model.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_codebook_size=4375 \
    'model.structure_encoder.embed_dim=5' \
    trainer.num_sanity_val_steps=2 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    logger.name=lg_fsq_sair_240_4375



