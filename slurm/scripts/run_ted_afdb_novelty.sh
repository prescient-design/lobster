#!/usr/bin/env bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/genume_novelty_rerun/%J_ted_afdb.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/genume_novelty_rerun/%J_ted_afdb.err
#SBATCH --job-name=ted_afdb_nov
#SBATCH -t 6:00:00

set -eo pipefail

cd /cv/home/lisanzas/lobster

AFDB_REPS="/cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb"
FOLDSEEK_BIN="/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin"

UNCOND_DIR="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional"

echo "=== AFDB novelty for GenUME-TED at $(date) ==="
echo "Uncond dir: ${UNCOND_DIR}"

uv run python scripts/analyze_tm_score_novelty.py \
    --uncond-dir "${UNCOND_DIR}" \
    --pdb-reps-pdb-dir "${AFDB_REPS}" \
    --foldseek-bin "${FOLDSEEK_BIN}" \
    --use-existing-clusters \
    --ref-label afdb

echo "=== Done at $(date) ==="
