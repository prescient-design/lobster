#!/usr/bin/env bash

#SBATCH --partition b200
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH -q preempt
#SBATCH -o slurm/logs/eval/%J_%x.out
#SBATCH --job-name=eval-protaux
#SBATCH -t 7-00:00:00

# srun hostname

nvidia-smi


source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

umask g+w

lobster_eval \
        --config-path /homefs/home/zadorozk/lobster/src/lobster/hydra_config/evaluation/ume-2 \
        --config-name amino_acid