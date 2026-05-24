#!/usr/bin/env bash
#SBATCH --partition=himem
#SBATCH --account=llm
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/convert_afdb_reps/%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/convert_afdb_reps/%A_%a.err
#SBATCH --job-name=convert_afdb_reps
#SBATCH -t 2:00:00
#SBATCH --array=0-19

set -eo pipefail

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/convert_afdb_reps

ARRAY_SIZE=20
OUTPUT_DIR="/cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb"

cd /cv/home/lisanzas/lobster

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Node: $(hostname)"

uv run python scripts/convert_afdb_cluster_reps_to_pdb.py \
    --output-dir "${OUTPUT_DIR}" \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    --array-size "${ARRAY_SIZE}"

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID completed at $(date) ==="
