#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_denovo_ted_cath/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_denovo_ted_cath/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_ted_cath_ssbal
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Only Training: PDB + AFDB + Denovo + TED + CATH (SS-balanced)
#
# SS-balanced run (Option C): strand-only and mixed clusters replicated up to 10x,
# helix-only clusters not upsampled. Expected SSE: H ~34% | S ~20% | C ~47% | H/S ~1.7
#
# Compare against baseline Job 6061624 (no SS balancing, H/S 3.39).
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_denovo_ted_cath_ss_balanced.sh

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_denovo_ted_cath/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_denovo_ted_cath
mkdir -p /cv/scratch/u/lisanzas/gen_ume_denovo_ted_cath/runs

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_gen_ume_denovo \
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
    logger.name=gen_ume_denovo-medium_pdb_afdb_denovo_ted_cath_ss_balanced \
    logger.project=lobster_gen_ume_denovo
