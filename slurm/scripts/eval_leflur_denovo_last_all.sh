#!/usr/bin/env bash
# Submit all 3 eval jobs in parallel for gen_ume_denovo last.ckpt
# Usage: bash slurm/scripts/eval_gen_ume_denovo_last_all.sh
#
# Copies the checkpoint to a timestamped snapshot before eval so it can be
# re-run later even if training overwrites last.ckpt.

CKPT_SOURCE="/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-08T17-09-23/last.ckpt"
RUN_TS="2026-03-08T17-09-23"  # must match the run dir name in CKPT_SOURCE
SNAPSHOT_DIR="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots"

EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
# Each eval run gets its own snapshot (timestamped) so we never overwrite previous copies
SNAPSHOT_CKPT="${SNAPSHOT_DIR}/gen_ume_denovo_last_${RUN_TS}_${EVAL_TIMESTAMP}.ckpt"

mkdir -p "${SNAPSHOT_DIR}"
if [ -f "${CKPT_SOURCE}" ]; then
    echo "Copying checkpoint to snapshot: ${SNAPSHOT_CKPT}"
    cp "${CKPT_SOURCE}" "${SNAPSHOT_CKPT}"
    CKPT="${SNAPSHOT_CKPT}"
else
    echo "WARNING: Checkpoint not found at ${CKPT_SOURCE}, using as-is"
    CKPT="${CKPT_SOURCE}"
fi
echo "Eval timestamp: ${EVAL_TIMESTAMP}"

cd /cv/home/lisanzas/lobster
sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${CKPT}" slurm/scripts/eval_gen_ume_denovo_last_forward.sh
sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${CKPT}" slurm/scripts/eval_gen_ume_denovo_last_inverse.sh
sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${CKPT}" slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
