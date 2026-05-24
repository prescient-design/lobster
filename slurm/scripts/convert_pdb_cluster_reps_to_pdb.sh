#!/usr/bin/env bash
#SBATCH --partition=himem
#SBATCH --account=llm
#SBATCH --qos=llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/convert_pdb_reps/%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/convert_pdb_reps/%A_%a.err
#SBATCH --job-name=convert_pdb_reps
#SBATCH -t 2:00:00
#SBATCH --array=0-99

# Convert PDB seqid40 cluster representatives from .pt to .pdb (100-way parallel)
# Output: /cv/scratch/u/lisanzas/pdb_seqid40_cluster_reps_pdb

set -eo pipefail

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/convert_pdb_reps

ARRAY_SIZE=100
OUTPUT_DIR="/cv/scratch/u/lisanzas/pdb_seqid40_cluster_reps_pdb"

cd /cv/home/lisanzas/lobster

echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID started at $(date) ==="
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

uv run python scripts/convert_pdb_cluster_reps_to_pdb.py \
    --output-dir "${OUTPUT_DIR}" \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    --array-size "${ARRAY_SIZE}"

echo ""
echo "=== Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID completed at $(date) ==="
