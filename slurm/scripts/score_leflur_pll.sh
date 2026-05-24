#!/usr/bin/env bash
# Score Gen-UME pseudo-likelihood (P1 + P2) on existing eval directories.
#
# Submits one B200 job per (eval-dir, ckpt) target. Defaults to scoring all three
# primary targets in parallel:
#   1) CAMEO forward-folding          -> default ckpt (epoch=17 best)
#   2) CAMEO inverse-folding          -> default ckpt (epoch=17 best)
#   3) SR-paired unconditional        -> val25-base SR snapshot ckpt
#
# Usage:
#   bash slurm/scripts/score_gen_ume_pll.sh                    # submit all 3
#   bash slurm/scripts/score_gen_ume_pll.sh forward            # submit just FF
#   bash slurm/scripts/score_gen_ume_pll.sh inverse            # submit just IF
#   bash slurm/scripts/score_gen_ume_pll.sh sr_paired          # submit just SR
#   bash slurm/scripts/score_gen_ume_pll.sh extra_uncond       # extend to TED-stoch SR-paired
#
# Scoring runs as a single sbatch with --export carrying EVAL_DIR / CKPT / TASK
# and re-execs the same script as the SBATCH worker (mode=run).

set -euo pipefail

DEFAULT_CKPT="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-06T15-30-31/epoch=17-step=6937-val_loss=0.8192.ckpt"
SR_BASE_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt"

EVAL_FF="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_cameo_forward_folding"
EVAL_IF="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_cameo_inverse_folding"
EVAL_SR_VAL25_BASE="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25_selfreflect_paired"
EVAL_SR_TED_STOCH="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10_selfreflect_paired"
EVAL_SR_TED_BIASED="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25_selfreflect_paired"
TED_CATH_SS_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"

# Conference-benchmark eval dirs (GenUME-base + GenUME-TED CAMEO + unconditional)
EVAL_BASE_FF="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-11T12-11-53_cameo_forward_folding"
EVAL_BASE_IF="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-11T12-11-53_cameo_inverse_folding"
EVAL_BASE_UC="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-11T12-11-53_unconditional"
EVAL_TED_FF="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_cameo_forward_folding"
EVAL_TED_IF="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_cameo_inverse_folding"
EVAL_TED_UC="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional"

K_DRAWS="${K_DRAWS:-32}"

###############################################################################
# Worker mode (executed by sbatch)
###############################################################################
if [ "${1:-}" = "--worker" ]; then
    : "${EVAL_DIR:?EVAL_DIR not set}"
    : "${CKPT:?CKPT not set}"
    : "${TASK:?TASK not set}"

    cd /cv/home/lisanzas/lobster
    echo "[worker] EVAL_DIR=${EVAL_DIR}"
    echo "[worker] CKPT=${CKPT}"
    echo "[worker] TASK=${TASK}"
    echo "[worker] K_DRAWS=${K_DRAWS}"
    echo "[worker] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  node=$(hostname)"

    if [ ! -d "${EVAL_DIR}" ]; then
        echo "ERROR: EVAL_DIR not found: ${EVAL_DIR}" >&2
        exit 1
    fi
    if [ ! -f "${CKPT}" ]; then
        echo "ERROR: CKPT not found: ${CKPT}" >&2
        exit 1
    fi

    uv run python scripts/score_gen_ume_pll.py \
        --eval-dir "${EVAL_DIR}" \
        --ckpt "${CKPT}" \
        --task "${TASK}" \
        --K "${K_DRAWS}" \
        --log-every 25
    exit $?
fi

###############################################################################
# Submit mode
###############################################################################
SUBMIT_TARGET="${1:-all}"

submit_job() {
    local job_name="$1"
    local eval_dir="$2"
    local ckpt="$3"
    local task="$4"

    if [ ! -d "${eval_dir}" ]; then
        echo "WARN: skipping ${job_name}: EVAL_DIR not found: ${eval_dir}" >&2
        return
    fi
    if [ ! -f "${ckpt}" ]; then
        echo "WARN: skipping ${job_name}: CKPT not found: ${ckpt}" >&2
        return
    fi

    mkdir -p /cv/scratch/u/lisanzas/slurm_logs/score_gen_ume_pll
    echo "Submitting ${job_name}: eval=$(basename ${eval_dir})  task=${task}"

    sbatch \
        --partition=ai4dd-b200 \
        --account=llm \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gres=gpu:b200:1 \
        --cpus-per-task=8 \
        --mem=128G \
        --time=4:00:00 \
        --qos=llm \
        --job-name="pll_${job_name}" \
        --output="/cv/scratch/u/lisanzas/slurm_logs/score_gen_ume_pll/%J_${job_name}.out" \
        --error="/cv/scratch/u/lisanzas/slurm_logs/score_gen_ume_pll/%J_${job_name}.err" \
        --export="ALL,EVAL_DIR=${eval_dir},CKPT=${ckpt},TASK=${task},K_DRAWS=${K_DRAWS}" \
        "$0" --worker
}

case "${SUBMIT_TARGET}" in
    all)
        submit_job "ff_cameo"     "${EVAL_FF}"           "${DEFAULT_CKPT}" "forward_folding"
        submit_job "if_cameo"     "${EVAL_IF}"           "${DEFAULT_CKPT}" "inverse_folding"
        submit_job "sr_val25base" "${EVAL_SR_VAL25_BASE}" "${SR_BASE_CKPT}" "unconditional"
        ;;
    forward|ff|forward_folding)
        submit_job "ff_cameo" "${EVAL_FF}" "${DEFAULT_CKPT}" "forward_folding"
        ;;
    inverse|if|inverse_folding)
        submit_job "if_cameo" "${EVAL_IF}" "${DEFAULT_CKPT}" "inverse_folding"
        ;;
    sr_paired|sr|val25)
        submit_job "sr_val25base" "${EVAL_SR_VAL25_BASE}" "${SR_BASE_CKPT}" "unconditional"
        ;;
    extra_uncond)
        submit_job "sr_ted_stoch"  "${EVAL_SR_TED_STOCH}"  "${TED_CATH_SS_CKPT}" "unconditional"
        submit_job "sr_ted_biased" "${EVAL_SR_TED_BIASED}" "${TED_CATH_SS_CKPT}" "unconditional"
        ;;
    base)
        submit_job "base_ff" "${EVAL_BASE_FF}" "${SR_BASE_CKPT}"      "forward_folding"
        submit_job "base_if" "${EVAL_BASE_IF}" "${SR_BASE_CKPT}"      "inverse_folding"
        submit_job "base_uc" "${EVAL_BASE_UC}" "${SR_BASE_CKPT}"      "unconditional"
        ;;
    ted)
        submit_job "ted_ff"  "${EVAL_TED_FF}"  "${TED_CATH_SS_CKPT}"  "forward_folding"
        submit_job "ted_if"  "${EVAL_TED_IF}"  "${TED_CATH_SS_CKPT}"  "inverse_folding"
        submit_job "ted_uc"  "${EVAL_TED_UC}"  "${TED_CATH_SS_CKPT}"  "unconditional"
        ;;
    conference|baselines)
        submit_job "base_ff" "${EVAL_BASE_FF}" "${SR_BASE_CKPT}"      "forward_folding"
        submit_job "base_if" "${EVAL_BASE_IF}" "${SR_BASE_CKPT}"      "inverse_folding"
        submit_job "base_uc" "${EVAL_BASE_UC}" "${SR_BASE_CKPT}"      "unconditional"
        submit_job "ted_ff"  "${EVAL_TED_FF}"  "${TED_CATH_SS_CKPT}"  "forward_folding"
        submit_job "ted_if"  "${EVAL_TED_IF}"  "${TED_CATH_SS_CKPT}"  "inverse_folding"
        submit_job "ted_uc"  "${EVAL_TED_UC}"  "${TED_CATH_SS_CKPT}"  "unconditional"
        ;;
    *)
        echo "Unknown target: ${SUBMIT_TARGET}" >&2
        echo "Usage: $0 [all|forward|inverse|sr_paired|extra_uncond|base|ted|conference]" >&2
        exit 2
        ;;
esac

echo "Done."
