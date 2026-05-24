#!/usr/bin/env bash
#SBATCH --partition=himem
#SBATCH --account=llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-99
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/analyze_sse/%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/analyze_sse/%A_%a.err
#SBATCH --job-name=analyze_sse
#SBATCH -q llm

# Secondary structure analysis - full de novo dataset (772k .pt files)
# Uses himem partition (same as denovo dataset processing)
# 100-way array: ~7.7k files per task
# Each task writes denovo_sse_index_shard_{task_id}.parquet
# After all complete, run: uv run python scripts/analyze_secondary_structure.py --merge-shards --output /cv/scratch/u/lisanzas/denovo_dataset/ume_dataset/denovo_sse_index.parquet

echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT)"
echo "Started at: $(date)"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/analyze_sse
cd /cv/home/lisanzas/lobster

uv run python scripts/analyze_secondary_structure.py \
    --denovo-only \
    --output /cv/scratch/u/lisanzas/denovo_dataset/ume_dataset/denovo_sse_index.parquet \
    --resume

echo "Completed at: $(date), Exit: $?"
