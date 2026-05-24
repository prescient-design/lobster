#!/usr/bin/env bash
# Run forward folding nsteps sweep: 50, 100, 200, 300
# Uses checkpoint snapshot from seed test, standard seed 54321
#
# Usage: bash slurm/scripts/eval_gen_ume_denovo_last_forward_nsteps_sweep.sh
#
# After all jobs complete, run:
#   python lobster/scripts/plot_forward_folding_nsteps_sweep.py --eval-timestamp EVAL_TS

SNAPSHOT_CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23.ckpt"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
SEED=54321

if [ ! -f "${SNAPSHOT_CKPT}" ]; then
    echo "ERROR: Checkpoint snapshot not found: ${SNAPSHOT_CKPT}"
    exit 1
fi

echo "Nsteps sweep: 50, 100, 200, 300"
echo "Checkpoint: ${SNAPSHOT_CKPT}"
echo "Seed: ${SEED}"
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo ""

cd /cv/home/lisanzas/lobster

for NSTEPS in 50 100 200 300; do
    echo "Submitting nsteps=${NSTEPS}..."
    sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${SNAPSHOT_CKPT}",SEED="${SEED}",NSTEPS="${NSTEPS}" \
        slurm/scripts/eval_gen_ume_denovo_last_forward.sh
done

echo ""
echo "Jobs submitted. After completion, plot with:"
echo "  cd /cv/home/lisanzas/lobster && python scripts/plot_forward_folding_nsteps_sweep.py --eval-timestamp ${EVAL_TIMESTAMP}"
