#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_medium_afdb/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_medium_afdb/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_pl_med_afdb_scratch
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Ligand Training Script - MEDIUM MODEL + AFDB SwissProt (FROM SCRATCH)
#
# Trains from scratch with AFDB SwissProt (198k structures)
# as additional protein-only training data:
#   - PDB: protein-only structures, 278k (backbone)
#   - AFDB SwissProt: protein-only structures, 198k (AlphaFold DB)
#   - GEOM: ligand conformers, 247k (with bond_matrix)
#   - PDBBind: protein-ligand complexes, 44k train / 5.5k val / 5.5k test (with bond_matrix)
#   - SAIR: protein-ligand complexes, 560k (with bond_matrix)
#
# Model: medium (~480M params)
# Effective batch size: 36 * 8 GPUs * 20 accumulate = 5760
# SE3 augmentation + ligand minimization enabled
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_protein_ligand_medium_afdb.sh

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
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium_afdb_scratch/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write access
umask g+w

# Create log and run directories if they don't exist
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_medium_afdb
mkdir -p /cv/scratch/u/lisanzas/gen_ume_protein_ligand_medium_afdb_scratch/runs

# Unset SLURM env vars that Lightning uses for auto-detection
# This lets Lightning handle multi-GPU via DDP within a single process
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Medium model from scratch with AFDB SwissProt data
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    data=structure_ligand_pdb_afdb_sair_bond \
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
    logger.name=gen_ume_protein_ligand-medium_afdb_scratch_bs36_lr1e-3 \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true
