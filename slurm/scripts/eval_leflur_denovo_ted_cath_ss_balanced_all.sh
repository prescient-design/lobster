#!/usr/bin/env bash
# Submit all 3 eval jobs for gen_ume_denovo_ted_cath SS-BALANCED (Option C)
# Run dir: 2026-03-14T15-41-36 (Job 6062191)
# Usage: bash slurm/scripts/eval_gen_ume_denovo_ted_cath_ss_balanced_all.sh

CKPT_SOURCE="/cv/scratch/u/lisanzas/gen_ume_denovo_ted_cath/runs/2026-03-14T15-41-36/last.ckpt"
RUN_TS="2026-03-14T15-41-36"
SNAPSHOT_DIR="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots"
EVAL_PREFIX="gen_ume_denovo_ted_cath_ss_balanced"

EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
SNAPSHOT_CKPT="${SNAPSHOT_DIR}/${EVAL_PREFIX}_${RUN_TS}_${EVAL_TIMESTAMP}.ckpt"

mkdir -p "${SNAPSHOT_DIR}"
if [ -f "${CKPT_SOURCE}" ]; then
    echo "Copying SS-balanced checkpoint to snapshot: ${SNAPSHOT_CKPT}"
    cp "${CKPT_SOURCE}" "${SNAPSHOT_CKPT}"
    CKPT="${SNAPSHOT_CKPT}"
else
    echo "WARNING: Checkpoint not found at ${CKPT_SOURCE}, using as-is"
    CKPT="${CKPT_SOURCE}"
fi
echo "Eval timestamp: ${EVAL_TIMESTAMP}"
echo "Eval prefix: ${EVAL_PREFIX}"

cd /cv/home/lisanzas/lobster
sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${CKPT}",EVAL_PREFIX="${EVAL_PREFIX}" slurm/scripts/eval_gen_ume_denovo_last_forward.sh
sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${CKPT}",EVAL_PREFIX="${EVAL_PREFIX}" slurm/scripts/eval_gen_ume_denovo_last_inverse.sh
sbatch --export=ALL,EVAL_TIMESTAMP="${EVAL_TIMESTAMP}",CKPT="${CKPT}",EVAL_PREFIX="${EVAL_PREFIX}" slurm/scripts/eval_gen_ume_denovo_last_unconditional.sh
