#!/bin/bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=16G
#SBATCH -o /cv/scratch/u/lisanzas/plinder_logs/predownload_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/plinder_logs/predownload_%A_%a.err
#SBATCH --job-name=plinder-dl
#SBATCH -t 24:00:00
#SBATCH --array=0-3

set -euo pipefail

PLINDER_ENV="/cv/scratch/u/lisanzas/uv_env/plinder/.venv/bin/activate"
SCRIPT="/cv/home/lisanzas/lobster/scripts/predownload_plinder_systems.py"
VALID_IDS="/cv/scratch/u/lisanzas/proteina-complexa/assets/data/plinder_valid_ids.txt"

TOTAL=$(wc -l < "$VALID_IDS")
CHUNK_SIZE=$(( (TOTAL + 3) / 4 ))
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ "$END_IDX" -gt "$TOTAL" ]; then END_IDX=$TOTAL; fi

echo "=== Predownload task $SLURM_ARRAY_TASK_ID: systems $START_IDX to $END_IDX ==="

source "$PLINDER_ENV"
export PLINDER_RELEASE=2024-06
export PLINDER_ITERATION=v2

python "$SCRIPT" \
    --valid-ids "$VALID_IDS" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --batch-size 500

echo "=== Done: predownload chunk $SLURM_ARRAY_TASK_ID ==="
