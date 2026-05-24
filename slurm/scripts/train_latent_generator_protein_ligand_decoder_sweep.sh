#!/usr/bin/env bash

# This script launches a Hydra multirun sweep for decoder hidden dimension optimization
# Sweeps over decoder dimensions (512, 768, 960, 1024) and learning rates (1e-4, 5e-4)
# Each hyperparameter combination will be submitted as a separate SLURM job
# DO NOT run this with sbatch - just run it directly: bash train_latent_generator_protein_ligand_decoder_sweep.sh

# Set up environment
source .venv/bin/activate

# Ensure required environment variables are set
export LOBSTER_RUNS_DIR="/data2/ume/latent_generator_/runs/"
export LOBSTER_DATA_DIR="/data2/ume/.cache2/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

# Run the multirun sweep
# Hydra will automatically submit 8 separate SLURM jobs:
# 4 decoder dimensions × 2 learning rates = 8 total runs
# decoder_dim controls both struc_token_dim and ligand_struc_token_dim together
lobster_train --multirun \
    experiment=train_latent_generator_protein_ligand_decoder_sweep \
    decoder_dim=512,768,960,1024 \
    model.optim.lr=1e-4,5e-4

echo ""
echo "=========================================="
echo "Decoder Dimension Sweep launched successfully!"
echo "=========================================="
echo "8 SLURM jobs submitted:"
echo "  - 4 decoder dimensions: 512, 768, 960, 1024"
echo "  - 2 learning rates: 1e-4, 5e-4"
echo "  - Each dimension tested with both learning rates"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "View outputs in:"
echo "  /data2/ume/latent_generator_/multirun/<timestamp>/"
echo ""
echo "Monitor in WandB:"
echo "  Project: lobster_latent_generator"
echo "  Group: latent_gen_decoder_sweep_<timestamp>"
echo "=========================================="

