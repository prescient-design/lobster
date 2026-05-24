#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/cudaretry_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/cudaretry_%A_%a.err
#SBATCH --job-name=cuda-retry
#SBATCH -t 05:00:00
#SBATCH --array=3,10,19,21,23,25,26,27,29,30,31,34,35,38,40,41,43,45,46,50,51,55,61,64,67,77,78,79,85,86,88,91,94,95,104,107,108,113,114,115,116,128,130,131,133,145,146,148,150,158,159,160,163,172,174,175,178,193,195,197,198,204,206,207,208,212,213,216,218,221,222,223,227,237,238,239,241,245,248,255,256,264,267,268,271,277,282,283,285,289,290,293,294,301,303,304,306,316,318,319,320,323,327,328,332,347,349,351,353,360,362,363,370,380,381,383,387,409,412,414,416,426,427,430,433,446,448,449,455,467,468,470,471,478,479,480,491,497

set -euo pipefail

PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
TASK_LIST="${PROTEINA_DIR}/configs/targets/plinder_task_names.txt"
CONFIG_NAME="search_ligand_binder_local_pipeline"

cd "$PROTEINA_DIR"
source .venv/bin/activate
source env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LINE_NUM=$(( SLURM_ARRAY_TASK_ID + 1 ))
TASK_NAME=$(sed -n "${LINE_NUM}p" "$TASK_LIST")

if [ -z "$TASK_NAME" ]; then
    echo "No task at line $LINE_NUM, exiting."
    exit 0
fi

# Reuse the same run name so skip-existing works if evaluation already completed
RUN_NAME="plinder_nonpass10_${TASK_NAME}"
EVAL_DIR="${PROTEINA_DIR}/evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"

if [ -d "$EVAL_DIR/monomer_metrics" ]; then
    echo "[SKIP] $TASK_NAME (evaluation complete at $EVAL_DIR)"
    exit 0
fi

echo "[CUDA-RETRY] $TASK_NAME (array=$SLURM_ARRAY_TASK_ID, 10 replicas, top 30)"

if complexa design "configs/${CONFIG_NAME}.yaml" \
    "++run_name=${RUN_NAME}" \
    "++generation.task_name=${TASK_NAME}" \
    "++ckpt_path=${PROTEINA_DIR}/ckpts" \
    ++ckpt_name=complexa_ligand.ckpt \
    "++autoencoder_ckpt_path=${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt" \
    "++generation.search.best_of_n.replicas=10" \
    "++generation.filter.filter_samples_limit=30"; then
    echo "[DONE] $TASK_NAME"
else
    echo "[FAIL] $TASK_NAME (exit=$?)"
    exit 1
fi
