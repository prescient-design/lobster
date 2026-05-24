#!/bin/bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 48
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/foldseek_cluster_%j.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/foldseek_cluster_%j.err
#SBATCH --job-name=foldseek-clust
#SBATCH -t 04:00:00

set -euo pipefail

mkdir -p /cv/scratch/u/lisanzas/slurm_logs

cd /cv/home/lisanzas/lobster

echo "=== Foldseek clustering for distillation + redesign ==="

uv run python scripts/cluster_distillation_redesign.py \
    --dataset both \
    --tmscore_threshold 0.5 \
    --threads ${SLURM_CPUS_PER_TASK}

echo "=== Done ==="
