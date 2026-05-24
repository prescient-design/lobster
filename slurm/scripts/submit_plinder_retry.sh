#!/bin/bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=32G
#SBATCH -o /cv/scratch/u/lisanzas/plinder_logs/retry_%A.out
#SBATCH -e /cv/scratch/u/lisanzas/plinder_logs/retry_%A.err
#SBATCH --job-name=plinder-retry
#SBATCH -t 04:00:00

set -euo pipefail

PLINDER_ENV="/cv/scratch/u/lisanzas/uv_env/plinder/.venv/bin/activate"
DL_SCRIPT="/cv/home/lisanzas/lobster/scripts/predownload_plinder_systems.py"
PROC_SCRIPT="/cv/home/lisanzas/lobster/scripts/process_plinder_dataset.py"
MISSING_IDS="/cv/scratch/u/lisanzas/plinder_missing_ids.txt"
OUTPUT_DIR="/cv/scratch/u/lisanzas/plinder_processed"
POSEBUSTERS="/cv/home/lisanzas/lobster/data/posebusters/posebusters_benchmark_set"

source "$PLINDER_ENV"
export PLINDER_RELEASE=2024-06
export PLINDER_ITERATION=v2

echo "=== Step 1: Re-download $(wc -l < "$MISSING_IDS") missing systems ==="
python "$DL_SCRIPT" \
    --valid-ids "$MISSING_IDS" \
    --batch-size 500

echo "=== Step 2: Re-process with skip-existing ==="
python "$PROC_SCRIPT" \
    --valid-ids "$MISSING_IDS" \
    --output-dir "$OUTPUT_DIR" \
    --max-protein-length 0 \
    --posebusters-dir "$POSEBUSTERS" \
    --skip-existing \
    --num-workers 8

echo "=== Done: retry complete ==="
