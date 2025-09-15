#!/usr/bin/env bash

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 96
#SBATCH --mem=128G
#SBATCH -q preempt
#SBATCH -o slurm/logs/data/%J_%x.out
#SBATCH --job-name=biopython
#SBATCH -t 14-00:00:00

# srun hostname

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

umask g+w

# srun -u --cpus-per-task 96 --cpu-bind=cores,verbose \
#     python analyze_biopython_distributions_mp.py  AMPLIFY --num-samples 1_000_000 --output biopython_distr --num-workers 80 --batch-size 1000

srun -u --cpus-per-task 96 --cpu-bind=cores,verbose \
    python analyze_biopython_scalers_mp.py AMPLIFY --num-samples 1_000_000 --output biopython_scalers --num-workers 80 --batch-size 1000