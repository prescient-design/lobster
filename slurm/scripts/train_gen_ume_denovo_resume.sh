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
#SBATCH --job-name=gen_ume_denovo
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# Gen-UME De Novo RESUME - from best checkpoint after NCCL timeout (job 5783245)
#
# Best checkpoint: epoch=15-step=6035-val_loss=1.3874.ckpt
# Job 5783245 failed at step ~7421 (Epoch 19) due to NCCL collective timeout (30 min).
# Job 5820392 failed at step ~8823 (Epoch 22) - same NCCL ALLREDUCE timeout.
#
# Mitigations: batch_size reduced 48->32 to avoid load imbalance / OOM; accumulate increased to keep effective batch
#
# Usage:
#   sbatch slurm/scripts/train_gen_ume_denovo_resume.sh

# Resume from best checkpoint (job 5820392 ran to step ~8823 before NCCL timeout)
#CHECKPOINT="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T03-52-41/epoch=17-step=6853-val_loss=1.0424.ckpt"
CHECKPOINT="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/last.ckpt"

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "Resuming from: ${CHECKPOINT}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
# NCCL_TIMEOUT (PyTorch uses own 30min default; this may not apply to c10d)
export NCCL_TIMEOUT=7200

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

uv run lobster_train \
    experiment=train_gen_ume_denovo \
    data=structure_pdb_afdb_denovo \
    "model.ckpt_path='${CHECKPOINT}'" \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=40 \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=24 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    logger.name=gen_ume_denovo-medium_pdb_afdb_denovo_resume
