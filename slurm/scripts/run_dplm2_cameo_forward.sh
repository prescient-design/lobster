#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_cameo_fwd.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_cameo_fwd.err
#SBATCH --job-name=dplm2_cameo_fwd
#SBATCH -t 4:00:00

set -eo pipefail
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval

bash /cv/home/lisanzas/dplm/run_dplm2_cameo_forward.sh
