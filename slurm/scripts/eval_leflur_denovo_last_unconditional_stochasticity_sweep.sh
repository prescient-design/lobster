#!/usr/bin/env bash
# Sweep over all combinations of stochasticity_seq and stochasticity_struc: 10, 20, 30, 40, 50, 60
# 6 x 6 = 36 jobs
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_last_unconditional_stochasticity_sweep.sh
#
# Optional: CKPT=... EVAL_TIMESTAMP=... to override defaults

CKPT="${CKPT:-/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt}"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

echo "Stochasticity sweep: seq, struc in {10, 20, 30, 40, 50, 60} (36 jobs)"
echo "Checkpoint: ${CKPT}"
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo ""

cd /cv/home/lisanzas/lobster

for STOCHASTICITY_SEQ in 10 20 30 40 50 60; do
    for STOCHASTICITY_STRUC in 10 20 30 40 50 60; do
        echo "Submitting seq=${STOCHASTICITY_SEQ} struc=${STOCHASTICITY_STRUC}..."
        sbatch --job-name="uncond_${STOCHASTICITY_SEQ}_${STOCHASTICITY_STRUC}" \
            --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${CKPT}",STOCHASTICITY_SEQ="${STOCHASTICITY_SEQ}",STOCHASTICITY_STRUC="${STOCHASTICITY_STRUC}" \
            slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
    done
done

echo ""
echo "36 jobs submitted."
