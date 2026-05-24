#!/usr/bin/env bash
# Temperature sweep for unconditional generation
# Two versions: stochasticity 10/10 and base (20/60)
# Grid: temperature_seq x temperature_struc = {0.1, 0.2, 0.3, 0.4, 0.5} x {0.1, 0.2, 0.3, 0.4, 0.5}
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_last_unconditional_temperature_sweep.sh
#
# Optional: CKPT=... EVAL_TIMESTAMP=... to override defaults

CKPT="${CKPT:-/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt}"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT}"
    exit 1
fi

TEMP_SEQ_VALUES=(0.1 0.2 0.3 0.4 0.5)
TEMP_STRUC_VALUES=(0.1 0.2 0.3 0.4 0.5)

echo "Temperature sweep: temp_seq, temp_struc in {0.1, 0.2, 0.3, 0.4, 0.5} (50 jobs total)"
echo "Checkpoint: ${CKPT}"
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo ""

cd /cv/home/lisanzas/lobster

echo "=== Version 1: stochasticity_seq=10, stochasticity_struc=10 ==="
for tseq in "${TEMP_SEQ_VALUES[@]}"; do
    for tstruc in "${TEMP_STRUC_VALUES[@]}"; do
        echo "Submitting: stoch=10/10 temp_seq=${tseq} temp_struc=${tstruc}"
        sbatch --job-name="uncond_s10_tseq${tseq}_tstruc${tstruc}" \
            --export=ALL,CKPT="${CKPT}",EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",STOCHASTICITY_SEQ=10,STOCHASTICITY_STRUC=10,TEMPERATURE_SEQ="${tseq}",TEMPERATURE_STRUC="${tstruc}" \
            slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
    done
done

echo ""
echo "=== Version 2: stochasticity_seq=20, stochasticity_struc=60 (base) ==="
for tseq in "${TEMP_SEQ_VALUES[@]}"; do
    for tstruc in "${TEMP_STRUC_VALUES[@]}"; do
        echo "Submitting: stoch=20/60 temp_seq=${tseq} temp_struc=${tstruc}"
        sbatch --job-name="uncond_sbase_tseq${tseq}_tstruc${tstruc}" \
            --export=ALL,CKPT="${CKPT}",EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",STOCHASTICITY_SEQ=20,STOCHASTICITY_STRUC=60,TEMPERATURE_SEQ="${tseq}",TEMPERATURE_STRUC="${tstruc}" \
            slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
    done
done

echo ""
echo "50 jobs submitted (25 stoch=10/10 + 25 stoch=20/60)."
