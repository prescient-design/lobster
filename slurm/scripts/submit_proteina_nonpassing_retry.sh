#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/nonpass_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/nonpass_%A_%a.err
#SBATCH --job-name=nonpass-retry
#SBATCH -t 05:00:00
#SBATCH --array=1,2,3,4,7,9,10,11,13,16,19,21,23,25,26,27,29,30,31,34,35,37,38,40,41,43,45,46,49,50,51,53,55,56,58,61,64,66,67,74,76,77,78,79,84,85,86,87,88,89,91,94,95,96,99,102,103,104,107,108,111,112,113,114,115,116,118,120,124,127,128,130,131,133,136,137,139,143,145,146,148,149,150,151,153,154,157,158,159,160,163,165,167,169,172,174,175,178,180,184,187,189,191,193,195,197,198,201,202,204,206,207,208,211,212,213,216,218,219,220,221,222,223,225,227,229,230,231,232,235,236,237,238,239,241,242,244,245,248,255,256,257,258,259,260,262,264,267,268,269,271,272,274,275,276,277,282,283,284,285,286,287,288,289,290,293,294,295,297,301,303,304,305,306,308,309,313,316,318,319,320,322,323,327,328,331,332,337,340,342,344,345,346,347,349,351,353,354,356,358,360,362,363,370,372,374,376,378,380,381,383,385,386,387,390,391,395,400,401,402,404,406,407,409,412,414,416,418,420,422,424,425,426,427,430,432,433,436,437,438,439,442,444,445,446,448,449,450,453,455,456,457,458,460,465,467,468,470,471,472,473,474,475,478,479,480,488,489,491,492,493,494,495,496,497

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

# Use a distinct run name so results don't collide with previous runs
RUN_NAME="plinder_nonpass10_${TASK_NAME}"
EVAL_DIR="${PROTEINA_DIR}/evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"

if [ -d "$EVAL_DIR/monomer_metrics" ]; then
    echo "[SKIP] $TASK_NAME (evaluation complete at $EVAL_DIR)"
    exit 0
fi

echo "[RETRY] $TASK_NAME (array=$SLURM_ARRAY_TASK_ID, 10 replicas, top 30)"

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
