#!/usr/bin/env bash

#SBATCH --partition b200
#SBATCH --no-requeue
#SBATCH --nodes 10
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --cpus-per-task 16
#SBATCH -o slurm/logs/train/%J_%x.out
#SBATCH -q preempt
#SBATCH --mem=256G
#SBATCH --job-name=neo-mini
#SBATCH -t 7-00:00:00

# srun hostname

nvidia-smi

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/amazon/ofi-nccl/lib64

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO

export LOBSTER_RUNS_DIR="s3://prescient-lobster/ume/runs"
export LOBSTER_DATA_DIR="/data2/ume/.cache2/"
export LOBSTER_USER=$(whoami) # CHANGE TO YOUR WANDB USERNAME IF NOT YOUR UNIXID
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write
# access for newly created files. Remove if not needed
umask g+w

srun -u --cpus-per-task 16 --cpu-bind=cores,verbose \
    lobster_train \
    experiment=neobert/train_neobert_mini \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES
