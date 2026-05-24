#!/usr/bin/env bash
# Submit two parallel SR-paired runs (TED-val25-base + val25-base) with
# generation.self_reflection.save_failed_attempts=true to capture initial
# sequences + backbones of every QC-rejected attempt for post-hoc ESMFold
# concordance analysis.
#
# Each run uses:
#   - use_esmfold_validation=true  (paired ESMFold metrics on accepted samples)
#   - save_failed_attempts=true    (write failed_self_reflection/{*.fasta,*.pdb}
#                                   and a single failed_self_reflection.csv)
#
# Output dirs (suffix _concordance):
#   /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional_seq20_struc60_biasV1.0_steps25_concordance
#   /cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25_concordance
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_sr_concordance.sh

set -euo pipefail

cd /cv/home/lisanzas/lobster

# Common SR overrides
SR_OVERRIDES="generation.self_reflection.use_esmfold_validation=true generation.self_reflection.save_failed_attempts=true"
LOGIT_BIAS="+generation.sequence_logit_bias={V:1.0} generation.sequence_logit_bias_steps=25 ${SR_OVERRIDES}"

# --- Run 1: TED-val25-base ---
TED_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
if [ ! -f "${TED_CKPT}" ]; then
    echo "ERROR: TED checkpoint not found: ${TED_CKPT}"
    exit 1
fi

echo "Submitting TED-val25-base SR concordance run..."
sbatch --job-name="ted_val25base_sr_conc" \
    --export="ALL,CONFIG_NAME=generate_unconditional_denovo_self_reflection,CKPT=${TED_CKPT},EVAL_TIMESTAMP=2026-03-18T12-20-59,EVAL_PREFIX=gen_ume_denovo_ted_cath_ss_balanced,STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,BIAS_SUFFIX=_biasV1.0_steps25_concordance,LOGIT_BIAS_OVERRIDE=${LOGIT_BIAS}" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

# --- Run 2: val25-base (matches the existing val25-base SR-paired run) ---
BASE_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt"
if [ ! -f "${BASE_CKPT}" ]; then
    echo "ERROR: val25-base checkpoint not found: ${BASE_CKPT}"
    exit 1
fi

echo "Submitting val25-base SR concordance run..."
sbatch --job-name="val25base_sr_conc" \
    --export="ALL,CONFIG_NAME=generate_unconditional_denovo_self_reflection,CKPT=${BASE_CKPT},EVAL_TIMESTAMP=2026-03-17T11-23-58,EVAL_PREFIX=gen_ume_denovo_last,STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,BIAS_SUFFIX=_biasV1.0_steps25_concordance,LOGIT_BIAS_OVERRIDE=${LOGIT_BIAS}" \
    slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh

echo "Both jobs submitted. Use 'squeue -u $(whoami)' to monitor."
