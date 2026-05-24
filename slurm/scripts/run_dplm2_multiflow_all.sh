#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_multiflow_all.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_multiflow_all.err
#SBATCH --job-name=dplm2_mf_all
#SBATCH -t 8:00:00

set -eo pipefail
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval

bash /cv/home/lisanzas/dplm/run_dplm2_multiflow_all.sh
