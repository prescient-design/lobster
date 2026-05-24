#!/usr/bin/env bash

# This script launches a Hydra multirun sweep for ligand token count optimization
# Fixed: decoder_dim=512, lr=5e-4
# Sweeps over ligand token counts (256, 512, 1024, 2048, 4096)
# Each configuration will be submitted as a separate SLURM job
# DO NOT run this with sbatch - just run it directly: bash train_latent_generator_protein_ligand_tokens_sweep.sh

# Set up environment
source .venv/bin/activate

# Ensure required environment variables are set
export LOBSTER_RUNS_DIR="/data2/ume/latent_generator_/runs/"
export LOBSTER_DATA_DIR="/data2/ume/.cache2/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

# Run the multirun sweep
# Hydra will automatically submit 5 separate SLURM jobs:
# 5 ligand token counts with fixed decoder dim (512) and lr (5e-4)
# ligand_tokens controls both quantizer.ligand_n_tokens and decoder ligand_struc_token_codebook_size
lobster_train --multirun \
    experiment=train_latent_generator_protein_ligand_tokens_sweep \
    ligand_tokens=256,512,1024,2048,4096

echo ""
echo "=========================================="
echo "Ligand Token Count Sweep launched successfully!"
echo "=========================================="
echo "5 SLURM jobs submitted:"
echo "  - Ligand token counts: 256, 512, 1024, 2048, 4096"
echo "  - Fixed decoder dimension: 512"
echo "  - Fixed learning rate: 5e-4"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "View outputs in:"
echo "  /data2/ume/latent_generator_/multirun/<timestamp>/"
echo ""
echo "Monitor in WandB:"
echo "  Project: lobster_latent_generator"
echo "  Group: latent_gen_ligand_tokens_sweep_<timestamp>"
echo "=========================================="

