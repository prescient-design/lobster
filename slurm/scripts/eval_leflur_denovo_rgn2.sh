#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_rgn2/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_rgn2/%J_%x.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_rgn2
#SBATCH -t 1-00:00:00
#SBATCH -q llm

# Gen-UME Forward Folding Evaluation on RGN2 dataset (orphan + denovo chains)

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_rgn2

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name experiment/research/generate_forward_folding_denovo_rgn2
