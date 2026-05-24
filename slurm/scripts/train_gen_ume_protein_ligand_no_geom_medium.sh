#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_no_geom_medium/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_no_geom_medium/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_pl_med_ng
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Ligand Training Script - MEDIUM MODEL, NO GEOM
#
# Trains the MEDIUM unified model (~480M params) on:
# - PDB: protein-only (49k clusters, seqid40)
# - AFDB SwissProt: protein-only (78k clusters)
# - PDBBind: protein-ligand (3.5k clusters, MMseqs2 40%)
# - SAIR: protein-ligand (2k clusters, MMseqs2 40%)
#
# balance_datasets: true → each dataset ~25% per epoch
# NO GEOM - protein-centric training with ligand interaction data
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_protein_ligand_no_geom_medium.sh

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=""
# Reduce OOM from fragmentation (log showed 122 GiB reserved but unallocated)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_medium/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_no_geom_medium
mkdir -p /cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_medium/runs

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

CHECKPOINT="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_medium/runs/2026-03-11T03-28-57/last.ckpt"
echo "Resuming from: ${CHECKPOINT}"

# Data: structure_ligand_pdb_afdb_sair_no_geom (PDB+AFDB+PDBBind+SAIR, balance_datasets, cluster files)
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand_no_geom \
    data=structure_ligand_pdb_afdb_sair_no_geom \
    "model.ckpt_path='${CHECKPOINT}'" \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=30 \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=22 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    logger.name=gen_ume_protein_ligand-medium_no_geom_bs30_balanced \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true

