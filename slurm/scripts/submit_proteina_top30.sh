#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/top30_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/top30_%A_%a.err
#SBATCH --job-name=proteina-top30
#SBATCH -t 05:00:00
#SBATCH --array=0-499

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

RUN_NAME="plinder_top30_${TASK_NAME}"
EVAL_DIR="${PROTEINA_DIR}/evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"

if [ -d "$EVAL_DIR/monomer_metrics" ]; then
    echo "[SKIP] $TASK_NAME (evaluation complete at $EVAL_DIR)"
    exit 0
fi

echo "[RUN] $TASK_NAME (array=$SLURM_ARRAY_TASK_ID, 100 replicas, top 30)"

if complexa design "configs/${CONFIG_NAME}.yaml" \
    "++run_name=${RUN_NAME}" \
    "++generation.task_name=${TASK_NAME}" \
    "++ckpt_path=${PROTEINA_DIR}/ckpts" \
    ++ckpt_name=complexa_ligand.ckpt \
    "++autoencoder_ckpt_path=${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt" \
    "++generation.search.best_of_n.replicas=100" \
    "++generation.filter.filter_samples_limit=30"; then
    echo "[DONE] $TASK_NAME"
else
    echo "[FAIL] $TASK_NAME (exit=$?)"
    exit 1
fi
