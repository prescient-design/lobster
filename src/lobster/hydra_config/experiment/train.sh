#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 16
# srun hostname

nvidia-smi
source ~/.bashrc
source ~/github/lobster/.venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
export WANDB_BASE_URL=https://genentech.wandb.io/
export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# wandb login --relogin --host https://genentech.wandb.io/

export TOKENIZERS_PARALLELISM=true

srun -u --cpus-per-task 8 --cpu-bind=cores,verbose \
lobster_train experiment=ume \
    logger.entity="$(whoami)" 

