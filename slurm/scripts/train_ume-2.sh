#!/usr/bin/env bash

#SBATCH --partition b200
#SBATCH --nodes 10
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --cpus-per-task 16
#SBATCH -o slurm/logs/train/%J_%x.out
#SBATCH -q preempt
#SBATCH --mem=256G
#SBATCH --job-name=ume-prot
#SBATCH -t 7-00:00:00


nvidia-smi

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/amazon/ofi-nccl/lib64

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

export LOBSTER_RUNS_DIR="s3://prescient-lobster/ume/runs" # CHANGE TO YOUR S3 BUCKET
export LOBSTER_DATA_DIR="/data2/ume/.cache2/" # CHANGE TO YOUR DATA DIRECTORY
export LOBSTER_USER=$(whoami) # CHANGE TO YOUR WANDB USERNAME IF NOT YOUR UNIXID
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# NCCL settings for distributed training stability
export NCCL_TIMEOUT=7200  # 2 hours timeout
export NCCL_IB_TIMEOUT=50
export NCCL_IB_RETRY_CNT=10

# Sets default permissions to allow group write
# access for newly created files. Remove if not needed
umask g+w

srun -u --cpus-per-task $SLURM_CPUS_PER_TASK --cpu-bind=cores,verbose \
    lobster_train \
    experiment=ume-2/amino_acid \
    data.num_workers=8 \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES
