#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --mem=200G
#SBATCH --job-name=genume-all-4n
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_all/multinode_%J.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_all/multinode_%J.err
#SBATCH -t 7-00:00:00

# Gen-UME Protein-Ligand Training — 4 nodes (32 GPUs)
#
# Effective batch size: 28 * 32 GPUs * 6 accumulate = 5,376
# (close to single-node 28 * 8 * 25 = 5,600)
#
# Resume from latest ALL config checkpoint.

set -euo pipefail

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NUM_NODES = ${SLURM_JOB_NUM_NODES}"
echo "SLURM_NTASKS = ${SLURM_NTASKS}"

# Network setup for multi-node NCCL
export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/amazon/ofi-nccl/lib64
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export NCCL_DEBUG=WARN
export NCCL_PROTO=Simple

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_all/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_all

CHECKPOINT="/cv/scratch/u/lisanzas/gen_ume_all/runs/2026-03-28T04-10-18/last.ckpt"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

srun -u --ntasks-per-node=1 \
    uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    data=structure_ligand_all \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=28 \
    data.num_workers=6 \
    trainer.devices=8 \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.accumulate_grad_batches=6 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    trainer.precision=bf16-mixed \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true \
    logger.name=gen_ume_all-medium_bs28_6nodes_lr1e-3 \
    "model.ckpt_path='${CHECKPOINT}'"
