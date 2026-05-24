#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/top30r_%A_%a.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/top30r_%A_%a.err
#SBATCH --job-name=top30-retry
#SBATCH -t 05:00:00
#SBATCH --array=103,130,143,153,164,165,166,167,168,169,170,171,172,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,267,268,269,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,366,367,368,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,394,395,396,401,402,403,407,408,409,410,411,412,413,414,415,419,420,421,425,426,427,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,484,485,486,493,494,495

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

echo "[RETRY] $TASK_NAME (array=$SLURM_ARRAY_TASK_ID, 100 replicas, top 30)"

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
