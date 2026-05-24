#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_denovo/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_denovo/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_denovo_large
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Only Training Script - LARGE MODEL + De Novo Synthetic Data
#
# Protein-only training with de novo synthetic structures added alongside PDB and AFDB:
#   - PDB: protein-only structures, 278k (backbone, clustered at 40% seq identity)
#   - AFDB SwissProt: protein-only structures, 198k (AlphaFold DB)
#   - De Novo: synthetic structures, 772k from 4 gen models × 5 lengths (100-500),
#     18,109 TM-score clusters, with alternative_sequences for sequence diversity
#
# Hyperparams for the LARGE model (~1.2B params):
#   - model_size=large
#   - batch_size=30 (lowered from 34 for stability)
#   - accumulate_grad_batches=30 (increased from 26 to maintain effective batch size)
#   - Effective batch size: 30 * 8 GPUs * 30 accumulate = 7200
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_denovo_large.sh

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Reduce OOM from fragmentation for large model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_denovo
mkdir -p /cv/scratch/u/lisanzas/gen_ume_denovo/runs

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

CHECKPOINT="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-08T14-28-42/last.ckpt"
echo "Resuming from: ${CHECKPOINT}"

uv run lobster_train \
    experiment=train_gen_ume_denovo \
    data=structure_pdb_afdb_denovo \
    "model.ckpt_path='${CHECKPOINT}'" \
    model.encoder_kwargs.model_size=large \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=30 \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=30 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    logger.name=gen_ume_denovo-large_pdb_afdb_denovo
