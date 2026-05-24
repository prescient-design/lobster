#!/bin/bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=32G
#SBATCH -o /cv/scratch/u/lisanzas/plinder_logs/%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/plinder_logs/%A_%a.err
#SBATCH --job-name=plinder-process
#SBATCH -t 02:00:00
#SBATCH --array=0-77

set -euo pipefail

PLINDER_ENV="/cv/scratch/u/lisanzas/uv_env/plinder/.venv/bin/activate"
SCRIPT="/cv/home/lisanzas/lobster/scripts/process_plinder_dataset.py"
VALID_IDS="/cv/scratch/u/lisanzas/proteina-complexa/assets/data/plinder_valid_ids.txt"
OUTPUT_DIR="/cv/scratch/u/lisanzas/plinder_processed"
POSEBUSTERS="/cv/home/lisanzas/lobster/data/posebusters/posebusters_benchmark_set"

TOTAL_SYSTEMS=$(wc -l < "$VALID_IDS")
CHUNK_SIZE=$(( (TOTAL_SYSTEMS + 77) / 78 ))  # 78 array tasks
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))

if [ "$END_IDX" -gt "$TOTAL_SYSTEMS" ]; then
    END_IDX=$TOTAL_SYSTEMS
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID: systems $START_IDX to $END_IDX ==="

mkdir -p /cv/scratch/u/lisanzas/plinder_logs

source "$PLINDER_ENV"
export PLINDER_RELEASE=2024-06
export PLINDER_ITERATION=v2

python "$SCRIPT" \
    --valid-ids "$VALID_IDS" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx "$START_IDX" \
    --end-idx "$END_IDX" \
    --max-protein-length 0 \
    --posebusters-dir "$POSEBUSTERS" \
    --skip-existing \
    --num-workers 8

echo "=== Done: chunk $SLURM_ARRAY_TASK_ID ==="
