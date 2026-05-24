#!/usr/bin/env bash
#SBATCH --partition himem
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=64G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/genume_novelty_rerun/%J.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/genume_novelty_rerun/%J.err
#SBATCH --job-name=novelty_rerun
#SBATCH -t 6:00:00

set -eo pipefail

cd /cv/home/lisanzas/lobster

PDB_REPS="/cv/scratch/u/lisanzas/pdb_seqid40_cluster_reps_pdb"
AFDB_REPS="/cv/scratch/u/lisanzas/afdb_swissprot_cluster_reps_pdb"
FOLDSEEK_BIN="/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin"

GENUME_DIRS=(
    "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-11T12-11-53_unconditional"
    "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-12T19-31-50_unconditional_seq10_struc10"
    "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-16T13-19-41_unconditional_seq10_struc10_tseq0.5_tstruc0.4"
    "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-16T22-46-28_unconditional_seq10_struc10_tseq0.5_tstruc0.4_biasV1.0_steps10"
    "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-16T22-46-28_unconditional_seq10_struc10_tseq0.5_tstruc0.4_biasV1.0_steps25"
    "/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-17T11-23-58_unconditional_seq20_struc60_biasV1.0_steps25"
)

for UNCOND_DIR in "${GENUME_DIRS[@]}"; do
    NAME=$(basename "${UNCOND_DIR}")
    echo "============================================================"
    echo "Processing: ${NAME}"
    echo "============================================================"

    # Clear stale novelty_analysis directory
    if [ -d "${UNCOND_DIR}/novelty_analysis" ]; then
        echo "Clearing stale novelty_analysis/"
        rm -rf "${UNCOND_DIR}/novelty_analysis"
    fi

    # Run 1: PDB + DeNovo novelty
    echo "--- PDB + DeNovo novelty at $(date) ---"
    uv run python scripts/analyze_tm_score_novelty.py \
        --uncond-dir "${UNCOND_DIR}" \
        --pdb-reps-pdb-dir "${PDB_REPS}" \
        --foldseek-bin "${FOLDSEEK_BIN}" \
        --use-existing-clusters \
        --ref-label pdb

    # Run 2: AFDB novelty (denovo is recomputed but identical)
    echo "--- AFDB novelty at $(date) ---"
    uv run python scripts/analyze_tm_score_novelty.py \
        --uncond-dir "${UNCOND_DIR}" \
        --pdb-reps-pdb-dir "${AFDB_REPS}" \
        --foldseek-bin "${FOLDSEEK_BIN}" \
        --use-existing-clusters \
        --ref-label afdb

    echo "Done with ${NAME} at $(date)"
    echo
done

echo "=== All novelty reruns complete at $(date) ==="
