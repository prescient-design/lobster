#!/usr/bin/env bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/genume_afdb_novelty/%J.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/genume_afdb_novelty/%J.err
#SBATCH --job-name=afdb_novelty
#SBATCH -t 2:00:00

# Run AFDB novelty for a single GenUME eval dir.
# Submit: sbatch --export=UNCOND_DIR=/path/to/eval slurm/scripts/run_genume_afdb_novelty.sh

set -eo pipefail

UNCOND_DIR="${UNCOND_DIR:?Must set UNCOND_DIR}"
AFDB_REPS="/cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb"

cd /cv/home/lisanzas/lobster

echo "=== AFDB novelty for $(basename ${UNCOND_DIR}) at $(date) ==="

uv run python scripts/analyze_tm_score_novelty.py \
    --uncond-dir "${UNCOND_DIR}" \
    --pdb-reps-pdb-dir "${AFDB_REPS}" \
    --foldseek-bin "/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin" \
    --use-existing-clusters \
    --ref-label afdb

echo "=== Done at $(date) ==="
