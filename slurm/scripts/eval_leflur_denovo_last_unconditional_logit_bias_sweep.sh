#!/usr/bin/env bash
# Logit bias sweep for unconditional generation -- Valine only
# 2D grid: bias strength x number of denoising steps with bias applied
#
# MODE selects the generation settings:
#   optimized (default): stoch=10/10, tseq=0.5, tstruc=0.4
#   base:                stoch=20/60, YAML-default temperatures (~0.273/~0.316)
#
# Usage:
#   bash slurm/scripts/eval_gen_ume_denovo_last_unconditional_logit_bias_sweep.sh          # optimized
#   MODE=base bash slurm/scripts/eval_gen_ume_denovo_last_unconditional_logit_bias_sweep.sh # base
#
# Optional: CKPT=... EVAL_TIMESTAMP=... to override defaults

CKPT="${CKPT:-/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt}"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
MODE="${MODE:-optimized}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

BIAS_STRENGTHS=(0.5 1.0 2.0 3.0)
BIAS_STEPS=(5 10 25 50 100)

if [ "${MODE}" = "base" ]; then
    STOCH_SEQ=20
    STOCH_STRUC=60
    TEMP_SEQ=""
    TEMP_STRUC=""
    MODE_LABEL="base (stoch=20/60, YAML-default temps)"
else
    STOCH_SEQ=10
    STOCH_STRUC=10
    TEMP_SEQ=0.5
    TEMP_STRUC=0.4
    MODE_LABEL="optimized (stoch=10/10, tseq=0.5, tstruc=0.4)"
fi

TOTAL=$((${#BIAS_STRENGTHS[@]} * ${#BIAS_STEPS[@]}))
echo "Valine logit bias sweep: ${#BIAS_STRENGTHS[@]} strengths x ${#BIAS_STEPS[@]} step counts = ${TOTAL} jobs"
echo "  Strengths: ${BIAS_STRENGTHS[*]}"
echo "  Steps:     ${BIAS_STEPS[*]}"
echo "  Mode:      ${MODE_LABEL}"
echo "Checkpoint: ${CKPT}"
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo ""

cd /cv/home/lisanzas/lobster

for bias in "${BIAS_STRENGTHS[@]}"; do
    for steps in "${BIAS_STEPS[@]}"; do
        BIAS_SUFFIX="_biasV${bias}_steps${steps}"
        LOGIT_BIAS_OVERRIDE="+generation.sequence_logit_bias={V:${bias}} generation.sequence_logit_bias_steps=${steps}"
        echo "Submitting: V bias=${bias}, steps=${steps} (suffix=${BIAS_SUFFIX})"

        EXPORT_VARS="ALL,CKPT=${CKPT},EVAL_TIMESTAMP=${EVAL_TIMESTAMP},STOCHASTICITY_SEQ=${STOCH_SEQ},STOCHASTICITY_STRUC=${STOCH_STRUC},BIAS_SUFFIX=${BIAS_SUFFIX},LOGIT_BIAS_OVERRIDE=${LOGIT_BIAS_OVERRIDE}"
        if [ -n "${TEMP_SEQ}" ] && [ -n "${TEMP_STRUC}" ]; then
            EXPORT_VARS="${EXPORT_VARS},TEMPERATURE_SEQ=${TEMP_SEQ},TEMPERATURE_STRUC=${TEMP_STRUC}"
        fi

        sbatch --job-name="uncond_bV${bias}_s${steps}" \
            --export="${EXPORT_VARS}" \
            slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
    done
done

echo ""
echo "${TOTAL} jobs submitted."
