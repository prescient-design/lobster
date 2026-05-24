#!/usr/bin/env bash
# Submit all 3 SR runs (TED-val25-base, val25-base, TED-stoch) with the SR
# forward-fold-TM QC threshold raised from the default 0.8334 to 0.9, to test
# whether a tighter precision gate yields more designable retained samples.
#
# Each run uses:
#   - generation.self_reflection.quality_control.min_tm_score_forward=0.9
#   - use_esmfold_validation=true   (paired ESMFold metrics on accepted samples)
#   - save_failed_attempts=true     (capture rejected attempts for concordance)
#
# Output dirs (suffix _sr_tm0p9):
#   /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25_sr_tm0p9
#   /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25_sr_tm0p9
#   /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq10_struc10_sr_tm0p9
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_sr_tm0p9.sh

set -euo pipefail

cd /cv/home/lisanzas/lobster

# Common SR overrides: paired ESMFold + save rejected attempts + TM gate at 0.9
SR_OVERRIDES="generation.self_reflection.use_esmfold_validation=true generation.self_reflection.save_failed_attempts=true generation.self_reflection.quality_control.min_tm_score_forward=0.9"

# --- Run 1: TED-val25-base (stoch=20/60, V=1.0 logit bias) ---
TED_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
if [ ! -f "${TED_CKPT}" ]; then
    echo "ERROR: TED-val25-base checkpoint not found: ${TED_CKPT}"
    exit 1
fi

TED_OVERRIDES="+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=25 ${SR_OVERRIDES}"
echo "Submitting TED-val25-base SR (TM>=0.9)..."
sbatch --job-name="ted_val25base_sr_tm0p9" \
    --export="ALL,CONFIG_NAME=generate_unconditional_denovo_self_reflection,CKPT=${TED_CKPT},EVAL_TIMESTAMP=2026-03-18T12-20-59,EVAL_PREFIX=gen_ume_denovo_ted_cath_ss_balanced,STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,BIAS_SUFFIX=_biasV1.0_steps25_sr_tm0p9,LOGIT_BIAS_OVERRIDE=${TED_OVERRIDES}" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# --- Run 2: val25-base (stoch=20/60, V=1.0 logit bias) ---
BASE_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt"
if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: val25-base checkpoint not found: ${BASE_CKPT}"
    exit 1
fi

BASE_OVERRIDES="+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=25 ${SR_OVERRIDES}"
echo "Submitting val25-base SR (TM>=0.9)..."
sbatch --job-name="val25base_sr_tm0p9" \
    --export="ALL,CONFIG_NAME=generate_unconditional_denovo_self_reflection,CKPT=${BASE_CKPT},EVAL_TIMESTAMP=2026-03-17T11-23-58,EVAL_PREFIX=gen_ume_denovo_last,STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,BIAS_SUFFIX=_biasV1.0_steps25_sr_tm0p9,LOGIT_BIAS_OVERRIDE=${BASE_OVERRIDES}" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# --- Run 3: TED-stoch (stoch=10/10, no logit bias) ---
STOCH_CKPT="${TED_CKPT}"  # same checkpoint as TED-val25-base, different sampling params

echo "Submitting TED-stoch SR (TM>=0.9)..."
sbatch --job-name="ted_stoch_sr_tm0p9" \
    --export="ALL,CONFIG_NAME=generate_unconditional_denovo_self_reflection,CKPT=${STOCH_CKPT},EVAL_TIMESTAMP=2026-03-18T12-20-59,EVAL_PREFIX=gen_ume_denovo_ted_cath_ss_balanced,STOCHASTICITY_SEQ=10,STOCHASTICITY_STRUC=10,BIAS_SUFFIX=_sr_tm0p9,LOGIT_BIAS_OVERRIDE=${SR_OVERRIDES}" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

echo ""
echo "All 3 jobs submitted with SR forward-fold-TM gate raised to 0.9."
echo "Use 'squeue -u $(whoami)' to monitor."
