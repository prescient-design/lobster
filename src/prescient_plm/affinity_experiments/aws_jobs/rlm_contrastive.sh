#!/usr/bin/env bash

#SBATCH --nodes 1
#SBATCH --partition gpu2
#SBATCH --gpus-per-node 4
#SBATCH -o ./logs/RLM-%J.out
#SBATCH --mem 0

srun hostname
srun nvidia-smi

source ~/.bashrc

micromamba activate /homefs/home/ismaia11/micromamba/envs/moe_esm2
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1

python main_contrastive.py -m +sweep=classification_rlm_regular.yaml
