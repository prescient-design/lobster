#!/usr/bin/env bash
# Forward-folding best-of-N PLL selection on full CAMEO (127 targets).
#
# Generates N=10 candidate forward-folds per target with the existing CAMEO
# benchmark hyperparameters (nsteps=200, temperature_seq=0.361,
# temperature_struc=0.220, stochasticity_struc=20), scores each candidate inline
# with the in-model PLL (seq + struc + joint), and writes per-candidate +
# per-target CSVs that the analysis script consumes.
#
# Three checkpoint variants are supported:
#   denovo  -> /cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/epoch=17-step=6937-val_loss=0.8192.ckpt
#              (older training run, originally used for PLL theory + correlation analysis)
#   base    -> conference-benchmark "GenUME-base"  (snapshot 2026-03-11T12-11-53)
#   ted     -> conference-benchmark "GenUME-TED"   (TED-CATH SS-balanced, snapshot 2026-03-18T12-20-59)
#
# Usage:
#   bash slurm/scripts/run_forward_fold_bestofN_pll.sh                # submit all 3 (denovo, base, ted)
#   bash slurm/scripts/run_forward_fold_bestofN_pll.sh denovo         # submit just the denovo variant
#   bash slurm/scripts/run_forward_fold_bestofN_pll.sh base           # submit just GenUME-base
#   bash slurm/scripts/run_forward_fold_bestofN_pll.sh ted            # submit just GenUME-TED
#   bash slurm/scripts/run_forward_fold_bestofN_pll.sh baselines      # submit base + ted (skip denovo, since denovo already ran)
#   bash slurm/scripts/run_forward_fold_bestofN_pll.sh --worker       # (sbatch internal)
#
# Env overrides (set before invoking submit):
#   N_CANDIDATES (default 10)
#   K_DRAWS      (default 32)
#   MAX_TARGETS  (default unlimited; useful for smoke runs)

set -euo pipefail

CKPT_DENOVO="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/epoch=17-step=6937-val_loss=0.8192.ckpt"
CKPT_BASE="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt"
CKPT_TED="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"

INPUTS_GLOB="/cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/*.pt"
OUTPUT_ROOT="/cv/scratch/u/lisanzas/evaluations"

N_CANDIDATES="${N_CANDIDATES:-10}"
K_DRAWS="${K_DRAWS:-32}"
MAX_TARGETS="${MAX_TARGETS:-}"

###############################################################################
# Worker mode (executed by sbatch)
###############################################################################
if [ "${1:-}" = "--worker" ]; then
    : "${CKPT:?CKPT not set}"
    : "${INPUTS:?INPUTS not set}"
    : "${OUTPUT_DIR:?OUTPUT_DIR not set}"
    : "${VARIANT:?VARIANT not set}"

    cd /cv/home/lisanzas/lobster

    echo "[worker] VARIANT=${VARIANT}"
    echo "[worker] CKPT=${CKPT}"
    echo "[worker] INPUTS=${INPUTS}"
    echo "[worker] OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[worker] N=${N_CANDIDATES}  K=${K_DRAWS}  MAX_TARGETS=${MAX_TARGETS:-(all)}"
    echo "[worker] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  node=$(hostname)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

    extra=()
    if [ -n "${MAX_TARGETS}" ]; then
        extra+=(--max-targets "${MAX_TARGETS}")
    fi

    uv run python scripts/forward_fold_bestofN_pll.py \
        --inputs "${INPUTS}" \
        --ckpt "${CKPT}" \
        --output-dir "${OUTPUT_DIR}" \
        --N "${N_CANDIDATES}" \
        --K "${K_DRAWS}" \
        "${extra[@]}"
    exit $?
fi

###############################################################################
# Submit mode
###############################################################################
SUBMIT_TARGET="${1:-all}"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/bestofN_ff_pll

submit_variant() {
    local variant="$1"
    local ckpt="$2"
    local suffix=""
    # Keep N=10 outputs at the original path; tag any other N to avoid clobbering.
    if [ "${N_CANDIDATES}" != "10" ]; then
        suffix="_N${N_CANDIDATES}"
    fi
    local output_dir="${OUTPUT_ROOT}/gen_ume_${variant}_cameo_bestofN_pll${suffix}"

    if [ ! -f "${ckpt}" ]; then
        echo "ERROR: CKPT not found for variant '${variant}': ${ckpt}" >&2
        exit 1
    fi

    mkdir -p "${output_dir}"
    echo "Submitting bestofN_ff_pll variant='${variant}' N=${N_CANDIDATES} K=${K_DRAWS} max_targets=${MAX_TARGETS:-(all)}"
    echo "  ckpt:   ${ckpt}"
    echo "  output: ${output_dir}"

    sbatch \
        --partition=ai4dd-b200 \
        --account=llm \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres=gpu:b200:1 \
        --cpus-per-task=8 \
        --mem=128G \
        --time=18:00:00 \
        --qos=llm \
        --job-name="bestofN_ff_pll_${variant}" \
        --output="/cv/scratch/u/lisanzas/slurm_logs/bestofN_ff_pll/%J_${variant}.out" \
        --error="/cv/scratch/u/lisanzas/slurm_logs/bestofN_ff_pll/%J_${variant}.err" \
        --export="ALL,VARIANT=${variant},CKPT=${ckpt},INPUTS=${INPUTS_GLOB},OUTPUT_DIR=${output_dir},N_CANDIDATES=${N_CANDIDATES},K_DRAWS=${K_DRAWS},MAX_TARGETS=${MAX_TARGETS}" \
        "$0" --worker
}

case "${SUBMIT_TARGET}" in
    all)
        submit_variant "denovo" "${CKPT_DENOVO}"
        submit_variant "base"   "${CKPT_BASE}"
        submit_variant "ted"    "${CKPT_TED}"
        ;;
    baselines|conf|conference)
        submit_variant "base"   "${CKPT_BASE}"
        submit_variant "ted"    "${CKPT_TED}"
        ;;
    denovo)
        submit_variant "denovo" "${CKPT_DENOVO}"
        ;;
    base|GenUME-base|genume-base)
        submit_variant "base"   "${CKPT_BASE}"
        ;;
    ted|GenUME-TED|genume-ted)
        submit_variant "ted"    "${CKPT_TED}"
        ;;
    *)
        echo "Unknown target: ${SUBMIT_TARGET}" >&2
        echo "Usage: $0 [all|baselines|denovo|base|ted]" >&2
        exit 2
        ;;
esac

echo "Done."
