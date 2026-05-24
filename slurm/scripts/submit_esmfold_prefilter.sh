#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 12
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31/esmfold_logs/prefilter_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31/esmfold_logs/prefilter_%A_%a.err
#SBATCH --job-name=esm-prefilt
#SBATCH -t 04:00:00
#SBATCH --array=0-199

set -euo pipefail

cd /cv/home/lisanzas/lobster

REDESIGNS_CSV="/cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31/plinder_ligandmpnn_redesigns.csv"
PLINDER_DIR="/cv/scratch/u/lisanzas/plinder_processed/train/"
OUTPUT_DIR="/cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31/esmfold_chunks"
N_CHUNKS=200

mkdir -p "$OUTPUT_DIR"
mkdir -p /cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31/esmfold_logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[START] chunk=${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}"

uv run python scripts/esmfold_prefilter_redesigns.py \
    --redesigns_csv "$REDESIGNS_CSV" \
    --plinder_data_dir "$PLINDER_DIR" \
    --output_csv "${OUTPUT_DIR}/chunk_${SLURM_ARRAY_TASK_ID}.csv" \
    --chunk_idx "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --max_rmsd 2.0 \
    --min_plddt 0.7

echo "[DONE] chunk=${SLURM_ARRAY_TASK_ID}"
