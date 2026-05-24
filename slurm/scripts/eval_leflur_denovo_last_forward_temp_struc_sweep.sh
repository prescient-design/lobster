#!/usr/bin/env bash
# Run forward folding temperature_struc sweep: 0.1, 0.15, 0.22, 0.3, 0.4
# Uses checkpoint snapshot, seed 54321, nsteps=500 (best from nsteps sweep)
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_last_forward_temp_struc_sweep.sh
#
# After all jobs complete, run:
#   uv run python scripts/plot_forward_folding_temp_struc_sweep.py --eval-timestamp EVAL_TS

SNAPSHOT_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23.ckpt"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
SEED=54321
NSTEPS=500  # best from nsteps sweep

if [ ! -f "${SNAPSHOT_CKPT}" ]; then
    echo "ERROR: Checkpoint snapshot not found: ${SNAPSHOT_CKPT}"
    exit 1
fi

echo "Temperature_struc sweep: 0.1, 0.15, 0.22, 0.3, 0.4"
echo "Checkpoint: ${SNAPSHOT_CKPT}"
echo "Seed: ${SEED}"
echo "Nsteps: ${NSTEPS}"
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo ""

cd /cv/home/lisanzas/lobster

for TEMPERATURE_STRUC in 0.1 0.15 0.22 0.3 0.4; do
    echo "Submitting temperature_struc=${TEMPERATURE_STRUC}..."
    sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${SNAPSHOT_CKPT}",SEED="${SEED}",NSTEPS="${NSTEPS}",TEMPERATURE_STRUC="${TEMPERATURE_STRUC}" \
        slurm/scripts/eval_gen_ume_denovo_last_forward.sh
done

echo ""
echo "Jobs submitted. After completion, plot with:"
echo "  cd /cv/home/lisanzas/lobster && uv run python scripts/plot_forward_folding_temp_struc_sweep.py --eval-timestamp ${EVAL_TIMESTAMP}"
