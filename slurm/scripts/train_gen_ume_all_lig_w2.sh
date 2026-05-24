#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH --mem=256G
#SBATCH --job-name=genume-all-ligw2
#SBATCH -t 7-00:00:00
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_all/ligw2_%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_all/ligw2_%j.err

# Sibling of train_gen_ume_all_resume.sh with ligand_struct_loss_weight=2.0
# Mirrors gen_ume_all-medium_bs28_lr1e-3_resume_0415: same starting checkpoint,
# same data, same lr/schedule/steps. Only difference: ligand_struct_loss_weight=2.0.
# Goal: improve ligand contact / ligand structure token fit.

set -euo pipefail

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_all/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_all

CHECKPOINT="/cv/scratch/u/lisanzas/gen_ume_all/runs/2026-04-15T19-53-26/last.ckpt"

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    data=structure_ligand_all \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    model.ligand_struct_loss_weight=2.0 \
    data.batch_size=28 \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=25 \
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
    logger.name=gen_ume_all-medium_bs28_lr1e-3_ligw2_resume_0415 \
    logger.project=lobster_gen_ume_protein_ligand \
    "model.ckpt_path='${CHECKPOINT}'"
