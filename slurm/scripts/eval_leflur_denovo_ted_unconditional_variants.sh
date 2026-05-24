#!/usr/bin/env bash
# Submit TED unconditional generation runs with the same parameter variants as the base model.
# Variants: stoch, temp, val, val25, val25-base
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_ted_unconditional_variants.sh

CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
EVAL_TIMESTAMP="2026-03-18T12-20-59"
EVAL_PREFIX="gen_ume_denovo_ted_cath_ss_balanced"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

echo "TED unconditional variants submission"
echo "Checkpoint: ${CKPT}"
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo "Eval prefix: ${EVAL_PREFIX}"
echo ""

cd /cv/home/lisanzas/lobster

# 1. TED-stoch: stoch=10/10, default temps
echo "Submitting TED-stoch (stoch=10/10)..."
sbatch --job-name="ted_stoch" \
    --export=ALL,CKPT="${CKPT}",EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",EVAL_PREFIX="${EVAL_PREFIX}",STOCHASTICITY_SEQ=10,STOCHASTICITY_STRUC=10 \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# 2. TED-temp: stoch=10/10, tseq=0.5, tstruc=0.4
echo "Submitting TED-temp (stoch=10/10, tseq=0.5, tstruc=0.4)..."
sbatch --job-name="ted_temp" \
    --export=ALL,CKPT="${CKPT}",EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",EVAL_PREFIX="${EVAL_PREFIX}",STOCHASTICITY_SEQ=10,STOCHASTICITY_STRUC=10,TEMPERATURE_SEQ=0.5,TEMPERATURE_STRUC=0.4 \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# 3. TED-val: stoch=10/10, tseq=0.5, tstruc=0.4, V=1.0, steps=10
echo "Submitting TED-val (stoch=10/10, tseq=0.5, tstruc=0.4, V=1.0, steps=10)..."
sbatch --job-name="ted_val" \
    --export="ALL,CKPT=${CKPT},EVAL_TIMESTAMP=${EVAL_TIMESTAMP},EVAL_PREFIX=${EVAL_PREFIX},STOCHASTICITY_SEQ=10,STOCHASTICITY_STRUC=10,TEMPERATURE_SEQ=0.5,TEMPERATURE_STRUC=0.4,BIAS_SUFFIX=_biasV1.0_steps10,LOGIT_BIAS_OVERRIDE=+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=10" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# 4. TED-val25: stoch=10/10, tseq=0.5, tstruc=0.4, V=1.0, steps=25
echo "Submitting TED-val25 (stoch=10/10, tseq=0.5, tstruc=0.4, V=1.0, steps=25)..."
sbatch --job-name="ted_val25" \
    --export="ALL,CKPT=${CKPT},EVAL_TIMESTAMP=${EVAL_TIMESTAMP},EVAL_PREFIX=${EVAL_PREFIX},STOCHASTICITY_SEQ=10,STOCHASTICITY_STRUC=10,TEMPERATURE_SEQ=0.5,TEMPERATURE_STRUC=0.4,BIAS_SUFFIX=_biasV1.0_steps25,LOGIT_BIAS_OVERRIDE=+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=25" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# 5. TED-val-base: stoch=20/60, default temps, V=1.0, steps=10
echo "Submitting TED-val-base (stoch=20/60, default temps, V=1.0, steps=10)..."
sbatch --job-name="ted_valbase" \
    --export="ALL,CKPT=${CKPT},EVAL_TIMESTAMP=${EVAL_TIMESTAMP},EVAL_PREFIX=${EVAL_PREFIX},STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,BIAS_SUFFIX=_biasV1.0_steps10,LOGIT_BIAS_OVERRIDE=+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=10" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# 6. TED-val25-base: stoch=20/60, default temps, V=1.0, steps=25
echo "Submitting TED-val25-base (stoch=20/60, default temps, V=1.0, steps=25)..."
sbatch --job-name="ted_val25base" \
    --export="ALL,CKPT=${CKPT},EVAL_TIMESTAMP=${EVAL_TIMESTAMP},EVAL_PREFIX=${EVAL_PREFIX},STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,BIAS_SUFFIX=_biasV1.0_steps25,LOGIT_BIAS_OVERRIDE=+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=25" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

echo ""
echo "6 unconditional variant jobs submitted."
