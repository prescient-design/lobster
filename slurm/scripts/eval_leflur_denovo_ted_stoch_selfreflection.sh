#!/usr/bin/env bash
# Submit GenUME-TED-stoch unconditional generation with self-reflection refinement enabled.
#
# Mirrors the existing SR-paired runs (TED-val25-base, val25-base) but with the
# TED-stoch generation hyperparameters (stochasticity_seq=10, stochasticity_struc=10,
# default temperatures, NO logit bias).
#
# Includes save_failed_attempts=true so we automatically get the per-attempt
# rejected sequence + initial backbone for post-hoc ESMFold concordance.
#
# Output dir:
#   /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10_selfreflect_paired/
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_ted_stoch_selfreflection.sh

set -euo pipefail

cd /cv/home/lisanzas/lobster

CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
EVAL_TIMESTAMP="2026-03-18T12-20-59"
EVAL_PREFIX="gen_ume_denovo_ted_cath_ss_balanced"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

SR_OVERRIDES="generation.self_reflection.use_esmfold_validation=true generation.self_reflection.save_failed_attempts=true"

echo "Submitting TED-stoch with SR (stoch=10/10, default temps, no logit bias, SR+paired+save_failed)..."
sbatch --job-name="ted_stoch_sr" \
    --export="ALL,CONFIG_NAME=generate_unconditional_denovo_self_reflection,CKPT=${CKPT},EVAL_TIMESTAMP=${EVAL_TIMESTAMP},EVAL_PREFIX=${EVAL_PREFIX},STOCHASTICITY_SEQ=10,STOCHASTICITY_STRUC=10,BIAS_SUFFIX=_selfreflect_paired,LOGIT_BIAS_OVERRIDE=${SR_OVERRIDES}" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
