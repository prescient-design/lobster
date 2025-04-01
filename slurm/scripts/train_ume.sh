#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH -o slurm/logs/%J.out
#SBATCH -t 24:00:00
# srun hostname

nvidia-smi

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

wandb login --host $WANDB_BASE_URL

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# Tokenizer calls prior in Ume callbacks prior to training
# cause issues. Disable if using callbacks, enable if not
export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write 
# access for newly created files. Remove if not needed
umask g+w 

srun -u --cpus-per-task 8 --cpu-bind=cores,verbose \
lobster_train experiment=train_ume ++data.datasets="['M320M', 'Calm', 'AMPLIFY', 'ZINC']"\
    logger.entity="$(whoami)" 


