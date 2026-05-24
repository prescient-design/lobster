#!/usr/bin/env bash
# Submit GenUME-TED-val25-base unconditional generation with self-reflection refinement enabled.
#
# Re-runs the same TED-val25-base configuration as
# slurm/scripts/eval_gen_ume_denovo_ted_unconditional_variants.sh (job 6) but
# with --config-name experiment/research/generate_unconditional_denovo_self_reflection so that
# generation.enable_self_reflection=true and the SR forward/inverse stages run
# with the GenUME-TED CAMEO benchmark hyperparameters.
#
# Output dir:
#   /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25_selfreflect/
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_ted_val25_base_selfreflection.sh

CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
EVAL_TIMESTAMP="2026-03-18T12-20-59"
EVAL_PREFIX="gen_ume_denovo_ted_cath_ss_balanced"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

cd /cv/home/lisanzas/lobster

echo "Submitting TED-val25-base with self-reflection (stoch=20/60, default temps, V=1.0, steps=25, SR enabled)..."
sbatch --job-name="ted_val25base_sr" \
    --export="ALL,CONFIG_NAME=generate_unconditional_denovo_self_reflection,CKPT=${CKPT},EVAL_TIMESTAMP=${EVAL_TIMESTAMP},EVAL_PREFIX=${EVAL_PREFIX},STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,BIAS_SUFFIX=_biasV1.0_steps25_selfreflect,LOGIT_BIAS_OVERRIDE=+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=25" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
