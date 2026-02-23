#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --account=llm
#SBATCH --array=1-50
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 4
#SBATCH -o /cv/scratch/u/lisanzas/sweeps/logs/%J_%x.out
#SBATCH -q preempt
#SBATCH --mem=256G
#SBATCH --job-name=pl_sweep
#SBATCH -t 2-00:00:00

# =============================================================================
# Protein-Ligand W&B Sweep SLURM Submission Script
# =============================================================================
# Usage:
#   1. Create sweep: wandb sweep wandb_sweep_config_protein_ligand_inverse_folding.yaml
#   2. Update SWEEP_ID below with the returned sweep ID
#   3. Submit: sbatch wandb_slurm.sh
# =============================================================================

# Set your sweep ID here (from wandb sweep command output)
SWEEP_ID="${SWEEP_ID:-f42gu2mv}"
WANDB_PROJECT="${WANDB_PROJECT:-lobster-wandb_sweeps}"

nvidia-smi

# Change to script directory
cd /cv/home/lisanzas/lobster/wandb_sweeps

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "SWEEP_ID = ${SWEEP_ID}"

export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/amazon/ofi-nccl/lib64

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Create log directory if it doesn't exist
mkdir -p /cv/scratch/u/lisanzas/sweeps/logs

# Run wandb agent with uv
srun -u --cpus-per-task $SLURM_CPUS_PER_TASK --cpu-bind=cores,verbose \
    uv run wandb agent "prescient-design/${WANDB_PROJECT}/${SWEEP_ID}"
