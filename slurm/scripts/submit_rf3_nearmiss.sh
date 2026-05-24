#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 8
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31/rf3_nearmiss_logs/rf3nm_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31/rf3_nearmiss_logs/rf3nm_%A_%a.err
#SBATCH --job-name=rf3-nearmiss
#SBATCH -t 08:00:00
#SBATCH --array=0-51

set -euo pipefail

LOBSTER_DIR="/cv/home/lisanzas/lobster"
PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
BASE_DIR="/cv/scratch/u/lisanzas/evaluations/plinder_redesign_2026-03-23T13-10-31"
REDESIGNS_CSV="${BASE_DIR}/plinder_redesigns_rf3_nearmiss_input.csv"
DATA_DIR="/cv/scratch/u/lisanzas/plinder_processed/train/"
RF3_CHUNK_DIR="${BASE_DIR}/rf3_nearmiss_chunks"
DESIGNS_PER_TASK=70

mkdir -p "$RF3_CHUNK_DIR"
mkdir -p "${BASE_DIR}/rf3_nearmiss_logs"

TASK_ID=${SLURM_ARRAY_TASK_ID}
START_IDX=$(( TASK_ID * DESIGNS_PER_TASK ))
END_IDX=$(( START_IDX + DESIGNS_PER_TASK ))
OUTPUT_CSV="${RF3_CHUNK_DIR}/rf3_chunk_${TASK_ID}.csv"
RF3_OUT="${BASE_DIR}/rf3_nearmiss_outputs/task_${TASK_ID}"

export RF3_CKPT_PATH="${PROTEINA_DIR}/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt"
export RF3_PATH="${PROTEINA_DIR}/.venv/bin/rf3"

cd "${PROTEINA_DIR}"
source .venv/bin/activate

echo "[START] task=${TASK_ID}, rows=${START_IDX}-${END_IDX}"

python "${LOBSTER_DIR}/scripts/run_rf3_eval_plinder.py" \
    --redesign_csv "${REDESIGNS_CSV}" \
    --data_dir "${DATA_DIR}" \
    --output_csv "${OUTPUT_CSV}" \
    --rf3_out_dir "${RF3_OUT}" \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX}

echo "[DONE] task=${TASK_ID}: ${OUTPUT_CSV}"
