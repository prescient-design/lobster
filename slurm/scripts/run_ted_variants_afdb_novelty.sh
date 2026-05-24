#!/usr/bin/env bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/genume_novelty_rerun/%J_ted_variants_afdb.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/genume_novelty_rerun/%J_ted_variants_afdb.err
#SBATCH --job-name=ted_var_afdb
#SBATCH -t 12:00:00

set -eo pipefail

cd /cv/home/lisanzas/lobster

AFDB_REPS="/cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb"
FOLDSEEK_BIN="/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin"
BASE="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_ted_cath_ss_balanced_ckpt_2026-03-18T12-20-59_unconditional"

VARIANTS=(
    "${BASE}_seq10_struc10"
    "${BASE}_seq10_struc10_tseq0.5_tstruc0.4"
    "${BASE}_seq10_struc10_tseq0.5_tstruc0.4_biasV1.0_steps10"
    "${BASE}_seq10_struc10_tseq0.5_tstruc0.4_biasV1.0_steps25"
    "${BASE}_seq20_struc60_biasV1.0_steps10"
    "${BASE}_seq20_struc60_biasV1.0_steps25"
)

for UNCOND_DIR in "${VARIANTS[@]}"; do
    NAME=$(basename "${UNCOND_DIR}")
    echo "============================================================"
    echo "AFDB novelty: ${NAME} at $(date)"
    echo "============================================================"

    uv run python scripts/analyze_tm_score_novelty.py \
        --uncond-dir "${UNCOND_DIR}" \
        --pdb-reps-pdb-dir "${AFDB_REPS}" \
        --foldseek-bin "${FOLDSEEK_BIN}" \
        --use-existing-clusters \
        --ref-label afdb

    echo "Done with ${NAME} at $(date)"
    echo
done

echo "=== All TED variant AFDB novelty complete at $(date) ==="
