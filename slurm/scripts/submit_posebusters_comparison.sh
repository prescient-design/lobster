#!/bin/bash
# Submit both Gen-UME and Proteina-Complexa generation + RF3 eval
# on the same 30 PoseBusters ligands for direct comparison

set -euo pipefail

LOBSTER_DIR="/cv/home/lisanzas/lobster"
PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
CKPT="${CKPT:-/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/checkpoints_all_latest_0406/last.ckpt}"
OUT_BASE="/cv/scratch/u/lisanzas/evaluations/posebusters_comparison"
LOG_DIR="/cv/scratch/u/lisanzas/slurm_logs/posebusters_comparison"

mkdir -p "$OUT_BASE" "$LOG_DIR"

echo "=============================================="
echo " PoseBusters Comparison: Gen-UME vs Proteina"
echo "=============================================="
echo "Checkpoint: ${CKPT}"
echo ""

# =============================================
# Job 1: Gen-UME generation + RF3 (single GPU)
# =============================================
GENUME_DIR="${OUT_BASE}/genume"

JOB_GENUME=$(sbatch --parsable --partition=ai4dd-b200 --account=llm --qos=llm \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=16 --mem=128G \
    -t 12:00:00 --job-name="pb-genume-rf3" \
    -o "${LOG_DIR}/genume_%j.out" \
    -e "${LOG_DIR}/genume_%j.err" \
    --wrap="
cd ${LOBSTER_DIR}

# Activate proteina env for RF3
source ${PROTEINA_DIR}/.venv/bin/activate

python scripts/conditioned_gen_best_with_rf3.py \
    --checkpoint '${CKPT}' \
    --output_dir '${GENUME_DIR}' \
    --num_samples 30 \
    --num_designs 5
echo 'DONE: Gen-UME + RF3'
")
echo "[1] Gen-UME + RF3 -> Job ${JOB_GENUME}"

# =============================================
# Job 2: Proteina-Complexa generation (array: 1 ligand per task)
# =============================================
PROTEINA_OUT="${OUT_BASE}/proteina"

JOB_PROTEINA=$(sbatch --parsable --partition=ai4dd-b200 --account=llm --qos=llm \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=24 --mem=128G \
    -t 05:00:00 --array=0-29 --job-name="pb-proteina" \
    -o "${LOG_DIR}/proteina_%A_%a.out" \
    -e "${LOG_DIR}/proteina_%A_%a.err" \
    --wrap='
PROTEINA_DIR='"${PROTEINA_DIR}"'
TASK_LIST="${PROTEINA_DIR}/configs/targets/posebusters/posebusters_task_names.txt"
CONFIG_NAME="search_ligand_binder_local_pipeline"

cd "$PROTEINA_DIR"
source .venv/bin/activate
source env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LINE_NUM=$(( SLURM_ARRAY_TASK_ID + 1 ))
TASK_NAME=$(sed -n "${LINE_NUM}p" "$TASK_LIST")

if [ -z "$TASK_NAME" ]; then
    echo "No task at line $LINE_NUM"
    exit 0
fi

RUN_NAME="posebusters_${TASK_NAME}"

echo "[START] $TASK_NAME (10 replicas)"

complexa design "configs/${CONFIG_NAME}.yaml" \
    "++run_name=${RUN_NAME}" \
    "++generation.task_name=${TASK_NAME}" \
    "++ckpt_path=${PROTEINA_DIR}/ckpts" \
    ++ckpt_name=complexa_ligand.ckpt \
    "++autoencoder_ckpt_path=${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt" \
    "++generation.search.best_of_n.replicas=10" \
    "++generation.filter.filter_samples_limit=30" \
    "++target_dict_cfg_yaml=${PROTEINA_DIR}/configs/targets/posebusters/posebusters_targets.yaml"

echo "[DONE] $TASK_NAME"
')
echo "[2] Proteina-Complexa -> Job ${JOB_PROTEINA} (array 0-29)"

# =============================================
# Job 3: RF3 eval on Proteina designs (after Job 2)
# =============================================
echo ""
echo "After Proteina jobs complete, run RF3 eval:"
echo "  # Filter passing designs"
echo "  uv run python scripts/filter_proteina_results.py ..."
echo "  # RF3 eval"
echo "  # (use the same pipeline as the distillation dataset)"
echo ""
echo "Output: ${OUT_BASE}/"
echo "  genume/   — Gen-UME generation + RF3"
echo "  proteina/ — Proteina-Complexa generation + RF3"
