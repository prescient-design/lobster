#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH -q preempt
#SBATCH --mem=256G
#SBATCH -o slurm/logs/%J.out
#SBATCH -t 00-01:00:00

# srun hostname

nvidia-smi

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export HF_HUB_CACHE=/data/bucket/freyn6/cache
export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

srun python examples/train_ume_grpo.py


