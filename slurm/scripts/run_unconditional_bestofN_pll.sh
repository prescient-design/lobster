#!/usr/bin/env bash
# Unconditional best-of-N PLL selection on LEFLUR-P-VAL (TED-val25-base).
#
# Mirrors slurm/scripts/run_forward_fold_bestofN_pll.sh but for unconditional
# generation. Generates N candidate (sequence, structure) pairs per (length,
# slot), scores each with the in-model PLL (seq + struc + joint sum + joint
# true), and ESMFolds each candidate sequence to compute self-consistency
# metrics (sc-TM and sc-RMSD vs the model's own decoded backbone, plus pLDDT).
#
# By default this submits ONE SLURM job PER LENGTH so the lengths run in
# parallel (the L=500 jobs dominate runtime; running them concurrently with
# the shorter lengths cuts wall-clock by ~5x vs a single sequential job).
#
# Usage:
#   bash slurm/scripts/run_unconditional_bestofN_pll.sh                  # submit per-length (5 jobs)
#   bash slurm/scripts/run_unconditional_bestofN_pll.sh single           # submit ONE sequential job over all lengths
#   bash slurm/scripts/run_unconditional_bestofN_pll.sh 100              # submit just length 100
#   bash slurm/scripts/run_unconditional_bestofN_pll.sh --worker         # (sbatch internal)
#
# Env overrides (set before invoking submit):
#   SLOTS        (default 100)        slots per length
#   N_CANDIDATES (default 30)         candidates per slot
#   K_DRAWS      (default 32)         PLL Monte-Carlo draws
#   LENGTHS      (default 100,200,300,400,500)
#   CKPT         (default LEFLUR-P-VAL TED-val25-base snapshot)
#   OUTPUT_DIR   (default /cv/scratch/u/lisanzas/evaluations/gen_ume_ted_lefp_val_bestofN_pll_unconditional_S<SLOTS>)

set -euo pipefail

CKPT_DEFAULT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
OUTPUT_ROOT="/cv/scratch/u/lisanzas/evaluations"

SLOTS="${SLOTS:-100}"
N_CANDIDATES="${N_CANDIDATES:-30}"
K_DRAWS="${K_DRAWS:-32}"
LENGTHS="${LENGTHS:-100,200,300,400,500}"
CKPT="${CKPT:-${CKPT_DEFAULT}}"
OUTPUT_DIR_DEFAULT="${OUTPUT_ROOT}/gen_ume_ted_lefp_val_bestofN_pll_unconditional_S${SLOTS}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_DEFAULT}}"

###############################################################################
# Worker mode (executed by sbatch)
###############################################################################
if [ "${1:-}" = "--worker" ]; then
    : "${CKPT:?CKPT not set}"
    : "${OUTPUT_DIR:?OUTPUT_DIR not set}"
    : "${WORKER_LENGTHS:?WORKER_LENGTHS not set}"
    : "${SLOTS:?SLOTS not set}"
    : "${N_CANDIDATES:?N_CANDIDATES not set}"

    cd /cv/home/lisanzas/lobster

    echo "[worker] CKPT=${CKPT}"
    echo "[worker] OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[worker] LENGTHS=${WORKER_LENGTHS}  SLOTS=${SLOTS}  N=${N_CANDIDATES}  K=${K_DRAWS}"
    echo "[worker] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  node=$(hostname)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    uv run python scripts/unconditional_bestofN_pll.py \
        --ckpt "${CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --N "${N_CANDIDATES}" \
        --slots "${SLOTS}" \
        --K "${K_DRAWS}" \
        --lengths "${WORKER_LENGTHS}"
    exit $?
fi

###############################################################################
# Submit mode
###############################################################################
SUBMIT_TARGET="${1:-per_length}"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/bestofN_uc_pll
mkdir -p "${OUTPUT_DIR}"

submit_one() {
    local tag="$1"
    local lengths_arg="$2"
    local time="$3"

    echo "Submitting bestofN_uc_pll tag='${tag}' lengths='${lengths_arg}' SLOTS=${SLOTS} N=${N_CANDIDATES} K=${K_DRAWS}"
    echo "  ckpt:   ${CKPT}"
    echo "  output: ${OUTPUT_DIR}"
    echo "  time:   ${time}"

    sbatch \
        --partition=ai4dd-b200 \
        --account=llm \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres=gpu:b200:1 \
        --cpus-per-task=8 \
        --mem=128G \
        --time="${time}" \
        --qos=llm \
        --job-name="bestofN_uc_pll_${tag}" \
        --output="/cv/scratch/u/lisanzas/slurm_logs/bestofN_uc_pll/%J_${tag}.out" \
        --error="/cv/scratch/u/lisanzas/slurm_logs/bestofN_uc_pll/%J_${tag}.err" \
        --export="ALL,CKPT=${CKPT},OUTPUT_DIR=${OUTPUT_DIR},WORKER_LENGTHS=${lengths_arg},SLOTS=${SLOTS},N_CANDIDATES=${N_CANDIDATES},K_DRAWS=${K_DRAWS}" \
        "$0" --worker
}

case "${SUBMIT_TARGET}" in
    per_length|all|"")
        # Per-length jobs run in parallel. Time scales superlinearly with L
        # (ESMFold dominates and is ~L^2 for memory + ~L^1.5 for compute).
        # Estimates from the 50-slot run (~5.5h total / 50 slots / 5 lengths
        # ≈ 13s/candidate at mean L; L=500 is ~3-4× the mean):
        #   L=100/200 -> ~3-5h for 100 slots × 30 cand
        #   L=300/400 -> ~6-10h
        #   L=500     -> ~12-18h
        # Pad each per-length job to 24h for safety.
        IFS=',' read -ra LARR <<< "${LENGTHS}"
        for L in "${LARR[@]}"; do
            submit_one "L${L}" "${L}" "24:00:00"
        done
        ;;
    single|sequential)
        # All lengths in one job. Worst-case ~50h sequential at SLOTS=100.
        submit_one "all" "${LENGTHS}" "48:00:00"
        ;;
    100|200|300|400|500)
        submit_one "L${SUBMIT_TARGET}" "${SUBMIT_TARGET}" "24:00:00"
        ;;
    *)
        echo "Unknown target: ${SUBMIT_TARGET}" >&2
        echo "Usage: $0 [per_length|single|100|200|300|400|500]" >&2
        exit 2
        ;;
esac

echo "Done."
