#!/usr/bin/env bash
# Run forward folding stochasticity_struc sweep: 10, 20, 30, 40
# Uses checkpoint snapshot, seed 54321, nsteps=500, temperature_struc=0.22 (best from prior sweeps)
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_last_forward_stoch_struc_sweep.sh
#
# After all jobs complete, run:
#   uv run python scripts/plot_forward_folding_stoch_struc_sweep.py --eval-timestamp EVAL_TS

SNAPSHOT_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23.ckpt"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
SEED=54321
NSTEPS=500
TEMPERATURE_STRUC=0.22

if [ ! -f "${SNAPSHOT_CKPT}" ]; then
    echo "ERROR: Checkpoint snapshot not found: ${SNAPSHOT_CKPT}"
    exit 1
fi

echo "Stochasticity_struc sweep: 10, 20, 30, 40"
echo "Checkpoint: ${SNAPSHOT_CKPT}"
echo "Seed: ${SEED}"
echo "Nsteps: ${NSTEPS}"
echo "Temperature_struc: ${TEMPERATURE_STRUC}"
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo ""

cd /cv/home/lisanzas/lobster

for STOCHASTICITY_STRUC in 10 20 30 40; do
    echo "Submitting stochasticity_struc=${STOCHASTICITY_STRUC}..."
    sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${SNAPSHOT_CKPT}",SEED="${SEED}",NSTEPS="${NSTEPS}",TEMPERATURE_STRUC="${TEMPERATURE_STRUC}",STOCHASTICITY_STRUC="${STOCHASTICITY_STRUC}" \
        slurm/scripts/eval_gen_ume_denovo_last_forward.sh
done

echo ""
echo "Jobs submitted. After completion, plot with:"
echo "  cd /cv/home/lisanzas/lobster && uv run python scripts/plot_forward_folding_stoch_struc_sweep.py --eval-timestamp ${EVAL_TIMESTAMP}"
