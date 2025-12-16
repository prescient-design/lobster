#!/bin/bash
#SBATCH --partition b200
#SBATCH --array=1-16
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH -o /data2/ume/gen_ume/slurm/logs/inference/%J_%x.out
#SBATCH -q preempt
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_hyp_param
#SBATCH -t 7-00:00:00

nvidia-smi

#source .venv/bin/activate
source /homefs/home/lisanzas/scratch/Develop/lobster/lobster_env/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/amazon/ofi-nccl/lib64

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO

export LOBSTER_RUNS_DIR="/data2/ume/gen_ume/runs/" #"s3://prescient-lobster/ume/runs" # CHANGE TO YOUR S3 BUCKET
export LOBSTER_DATA_DIR="/data2/ume/.cache2/" # CHANGE TO YOUR DATA DIRECTORY
export LOBSTER_USER=$(whoami) # CHANGE TO YOUR WANDB USERNAME IF NOT YOUR UNIXID
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

srun -u --cpus-per-task $SLURM_CPUS_PER_TASK --cpu-bind=cores,verbose wandb agent prescient-design/lobster-wandb_sweeps/b7wlmyg8