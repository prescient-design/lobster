#!/usr/bin/env bash
# Regenerate FF NLL-picked candidates per CAMEO target and Kabsch-align
# them onto GT for visualization.
#
# Submits a single GPU job (~15 min for 127 targets) that runs
# `scripts/regen_ff_nll_picked_aligned.py` on the TED checkpoint with the
# struc_pll picker.

set -euo pipefail

CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
INPUTS_GLOB="/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt"
CANDIDATES="/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N30/bestofN_ff_candidates_20260501T025401.csv"
SUMMARY="/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_bestofN_pll_N30/bestofN_ff_summary_20260501T025401.csv"
OUTPUT_DIR="/cv/scratch/u/lisanzas/evaluations/gen_ume_ted_cameo_ff_struc_pll_aligned"
PICKER="${PICKER:-struc_pll_pick_idx}"
MAX_TARGETS="${MAX_TARGETS:-}"

if [ "${1:-}" = "--worker" ]; then
    cd /cv/home/lisanzas/lobster
    echo "[worker] PICKER=${PICKER}"
    echo "[worker] OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[worker] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  node=$(hostname)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    extra=()
    if [ -n "${MAX_TARGETS}" ]; then extra+=(--max-targets "${MAX_TARGETS}"); fi

    uv run python scripts/regen_ff_nll_picked_aligned.py \
        --candidates "${CANDIDATES}" \
        --summary    "${SUMMARY}" \
        --inputs     "${INPUTS_GLOB}" \
        --ckpt       "${CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --picker     "${PICKER}" \
        "${extra[@]}"

    uv run python scripts/render_ff_overlays.py --root "${OUTPUT_DIR}"
    exit $?
fi

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/regen_ff_nll
mkdir -p "${OUTPUT_DIR}"

sbatch \
    --partition=ai4dd-b200 \
    --account=llm \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres=gpu:b200:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=02:00:00 \
    --qos=llm \
    --job-name="regen_ff_nll_${PICKER}" \
    --output="/cv/scratch/u/lisanzas/slurm_logs/regen_ff_nll/%J_${PICKER}.out" \
    --error="/cv/scratch/u/lisanzas/slurm_logs/regen_ff_nll/%J_${PICKER}.err" \
    --export="ALL,PICKER=${PICKER},MAX_TARGETS=${MAX_TARGETS}" \
    "$0" --worker

echo "Submitted. Logs: /cv/scratch/u/lisanzas/slurm_logs/regen_ff_nll/"
echo "Output: ${OUTPUT_DIR}"
