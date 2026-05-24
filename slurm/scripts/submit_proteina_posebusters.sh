#!/bin/bash
# Proteina-Complexa single-pass generation on 30 PoseBusters ligands
# 5 designs per ligand (matching Gen-UME), then RF3 eval
#
# Uses the posebusters targets merged into the main target YAML.
#
# Usage:
#   bash slurm/scripts/submit_proteina_posebusters.sh

set -euo pipefail

PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
LOBSTER_DIR="/cv/home/lisanzas/lobster"
OUT_BASE="/cv/scratch/u/lisanzas/evaluations/posebusters_comparison/proteina"
LOG_DIR="/cv/scratch/u/lisanzas/slurm_logs/posebusters_comparison"
TASK_LIST="${PROTEINA_DIR}/configs/targets/posebusters/posebusters_task_names.txt"

mkdir -p "$OUT_BASE" "$LOG_DIR"

echo "=== Proteina-Complexa PoseBusters Generation ==="
echo "  30 ligands × 5 designs = 150 total"
echo "  Algorithm: single-pass (no search/filtering)"
echo ""

# Write the per-task script
TASK_SCRIPT="${OUT_BASE}/run_proteina_task.sh"
cat > "${TASK_SCRIPT}" << 'TASKEOF'
#!/bin/bash
set -euo pipefail

PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
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

echo "[START] $TASK_NAME (single-pass × 5)"

for DESIGN_IDX in 0 1 2 3 4; do
    DESIGN_RUN="${RUN_NAME}_d${DESIGN_IDX}"
    echo "  Design ${DESIGN_IDX}/5: ${DESIGN_RUN}"

    complexa generate "configs/${CONFIG_NAME}.yaml" \
        "++run_name=${DESIGN_RUN}" \
        "++generation.task_name=${TASK_NAME}" \
        "++ckpt_path=${PROTEINA_DIR}/ckpts" \
        ++ckpt_name=complexa_ligand.ckpt \
        "++autoencoder_ckpt_path=${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt" \
        "++generation.search.algorithm=single-pass" \
        "generation/targets=posebusters_targets" \
    || echo "  Design ${DESIGN_IDX} failed"
done

echo "[DONE] $TASK_NAME (5 designs)"
TASKEOF
chmod +x "${TASK_SCRIPT}"

# Submit array job
sbatch --partition=ai4dd-b200 --account=llm --qos=llm \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=24 --mem=128G \
    -t 02:00:00 --array=0-29 --job-name="pb-proteina" \
    -o "${LOG_DIR}/proteina_%A_%a.out" \
    -e "${LOG_DIR}/proteina_%A_%a.err" \
    "${TASK_SCRIPT}"

echo "Submitted 30 Proteina-Complexa generation jobs"
