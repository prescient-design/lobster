#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/sweep_gen_ume_pl_lr_bs/%A_%a_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/sweep_gen_ume_pl_lr_bs/%A_%a_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=sweep_ume_pl
#SBATCH -t 2-00:00:00
#SBATCH -q llm
#SBATCH --array=0-11

# Gen-UME Protein-Ligand LR & Batch Size Sweep
# 
# Grid search over:
# - Learning rates: [5e-5, 1e-4, 2e-4, 5e-4]
# - Accumulate grad batches: [20, 40, 80] (effective BS: 1280, 2560, 5120)
#
# Total: 4 x 3 = 12 configurations
#
# Usage:
#   sbatch slurm/scripts/sweep_gen_ume_protein_ligand_lr_bs.sh

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_pl_sweep_lr_bs/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true

umask g+w
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/sweep_gen_ume_pl_lr_bs

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Define sweep parameters
LR_VALUES=(5e-5 1e-4 2e-4 5e-4)
ACCUM_VALUES=(20 40 80)

# Calculate indices
NUM_LR=${#LR_VALUES[@]}
NUM_ACCUM=${#ACCUM_VALUES[@]}

LR_IDX=$((SLURM_ARRAY_TASK_ID / NUM_ACCUM))
ACCUM_IDX=$((SLURM_ARRAY_TASK_ID % NUM_ACCUM))

LR=${LR_VALUES[$LR_IDX]}
ACCUM=${ACCUM_VALUES[$ACCUM_IDX]}
EFFECTIVE_BS=$((8 * 8 * ACCUM))

echo "Running config: LR=${LR}, ACCUM=${ACCUM}, Effective BS=${EFFECTIVE_BS}"

# Run training with sweep parameters
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand_no_geom \
    data.num_workers=8 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    trainer.accumulate_grad_batches=${ACCUM} \
    trainer.max_steps=20000 \
    trainer.val_check_interval=500 \
    model.lr=${LR} \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    logger.name=sweep_lr${LR}_bs${EFFECTIVE_BS} \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true

