#!/usr/bin/env bash

# Gen-UME Protein-Ligand Batch Size & Learning Rate Sweep (LARGE model)
#
# This script launches a Hydra multirun sweep for batch size and learning rate optimization
# using the LARGE model size.
#
# Sweeps over:
#   - Batch sizes: 28, 32, 36 per GPU
#   - Learning rates: 3e-4, 5e-4, 1e-3, 2e-3
#
# Total: 3 batch sizes × 4 learning rates = 12 SLURM jobs
#
# DO NOT run this with sbatch - run directly:
#   cd /cv/home/lisanzas/lobster
#   bash slurm/scripts/train_gen_ume_protein_ligand_batch_lr_sweep.sh
#
# NOTE: SE3 augmentation now uses proper rotations (no reflections) after fix to _kinematics.py
# NOTE: Medium model: bs=36 worked, bs=40 OOM, bs>=48 failed silently (OOM)
# NOTE: Large model needs smaller batch sizes than medium
#
# Effective batch sizes (with 8 GPUs × 20 accumulate):
#   - bs=28: 28 × 8 × 20 = 4480
#   - bs=32: 32 × 8 × 20 = 5120
#   - bs=36: 36 × 8 × 20 = 5760

set -e

# Change to lobster directory
cd /cv/home/lisanzas/lobster
echo "Working directory: $(pwd)"

# Set up environment variables
export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export WANDB_INSECURE_DISABLE_SSL=true

# Create log directory
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand

echo ""
echo "=========================================="
echo "Gen-UME Batch Size & LR Sweep (LARGE)"
echo "=========================================="
echo ""

# Run the multirun sweep
# Hydra will automatically submit separate SLURM jobs for each combination
uv run lobster_train --multirun \
    experiment=train_gen_ume_protein_ligand_batch_lr_sweep \
    model.encoder_kwargs.model_size=large \
    data.batch_size=28,32,36 \
    model.lr=3e-4,5e-4,1e-3,2e-3

echo ""
echo "=========================================="
echo "Sweep launched successfully!"
echo "=========================================="
echo ""
echo "12 SLURM jobs submitted (LARGE model):"
echo "  - 3 batch sizes: 28, 32, 36 (per GPU)"
echo "  - 4 learning rates: 3e-4, 5e-4, 1e-3, 2e-3"
echo ""
echo "Effective batch sizes (with 8 GPUs × 20 accumulate):"
echo "  - bs=28: 4480"
echo "  - bs=32: 5120"
echo "  - bs=36: 5760"
echo ""
echo "Note: medium model bs=40 was OOM, large needs smaller batches"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View outputs in:"
echo "  /cv/scratch/u/lisanzas/gen_ume_protein_ligand/multirun/<timestamp>/"
echo ""
echo "Monitor in WandB:"
echo "  Project: lobster_gen_ume_protein_ligand"
echo "  Group: gen_ume_batch_lr_sweep_<timestamp>"
echo "=========================================="

