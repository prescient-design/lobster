#!/usr/bin/env bash

#SBATCH --partitsion gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH -o slurm/logs/%J.out
#SBATCH -t 00-24:00:00

# srun hostname

nvidia-smi

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export LOBSTER_DATA_DIR="/data/lobster/ume/data"
export LOBSTER_RUNS_DIR="/data/lobster/ume/runs"
export LOBSTER_USER=$(whoami) # CHANGE TO YOUR WANDB USERNAME IF NOT YOUR UNIXID
export WANDB_BASE_URL=https://genentech.wandb.io

# Tokenizer calls prior in Ume callbacks prior to training
# cause issues. Disable if using callbacks, enable if not
export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write 
# access for newly created files. Remove if not needed
umask g+w 

srun -u --cpus-per-task 8 --cpu-bind=cores,verbose \
lobster_train experiment=train_ume


