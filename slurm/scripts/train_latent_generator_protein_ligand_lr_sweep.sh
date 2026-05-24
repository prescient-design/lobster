#!/usr/bin/env bash

# This script launches a Hydra multirun sweep for learning rate optimization
# Each hyperparameter combination will be submitted as a separate SLURM job
# DO NOT run this with sbatch - just run it directly: bash train_latent_generator_protein_ligand_lr_sweep.sh

# Set up environment
source .venv/bin/activate

# Ensure required environment variables are set
export LOBSTER_RUNS_DIR="/data2/ume/latent_generator_/runs/"
export LOBSTER_DATA_DIR="/data2/ume/.cache2/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

# Run the multirun sweep
# Hydra will automatically submit 4 separate SLURM jobs (one per learning rate)
lobster_train --multirun \
    experiment=train_latent_generator_protein_ligand_slurm \
    model.optim.lr=5e-3,1e-3,5e-4,1e-4

echo ""
echo "=========================================="
echo "Sweep launched successfully!"
echo "=========================================="
echo "4 SLURM jobs submitted (one per learning rate)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "View outputs in:"
echo "  multirun/<timestamp>/"
echo ""
echo "Monitor in WandB:"
echo "  Project: lobster_latent_generator"
echo "  Group: latent_gen_lr_sweep_<timestamp>"
echo "=========================================="

