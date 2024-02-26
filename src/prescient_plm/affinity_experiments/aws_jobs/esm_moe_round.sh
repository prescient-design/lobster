#!/usr/bin/env bash

#SBATCH --nodes 1
#SBATCH --partition gpu2
#SBATCH --gpus-per-node 8
#SBATCH -o ./logs/esm-%J.out
#SBATCH --mem 0

srun hostname
srun nvidia-smi

source ~/.bashrc

micromamba activate /homefs/home/ismaia11/micromamba/envs/prescient-plm
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1

python main.py -m +sweep=classification_esm2_150M_moe_round_regular.yaml
