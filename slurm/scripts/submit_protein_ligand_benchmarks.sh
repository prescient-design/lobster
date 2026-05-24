#!/usr/bin/env bash
set -euo pipefail

# Master script: submit all 4 protein-ligand evaluation SLURM jobs and
# print instructions for chained co-folding + merging.
#
# Usage:
#   bash slurm/scripts/submit_protein_ligand_benchmarks.sh
#
# Override checkpoint:
#   CKPT=/path/to/ckpt bash slurm/scripts/submit_protein_ligand_benchmarks.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap/"
export EVAL_TIMESTAMP

echo "=============================================="
echo " Protein-Ligand Benchmark Suite"
echo " Timestamp: ${EVAL_TIMESTAMP}"
echo "=============================================="
echo ""

# --- Phase 1: Submit evaluation jobs ---

JOB_LMPNN=$(sbatch --parsable "${SCRIPT_DIR}/eval_ligandmpnn_baseline_inverse_folding.sh")
echo "[1/4] LigandMPNN baseline inverse folding  -> Job ${JOB_LMPNN}"

JOB_IF=$(sbatch --parsable "${SCRIPT_DIR}/eval_gen_ume_protein_ligand_inverse_folding.sh")
echo "[2/4] Gen-UME inverse folding               -> Job ${JOB_IF}"

JOB_FF=$(sbatch --parsable "${SCRIPT_DIR}/eval_gen_ume_protein_ligand_forward_folding.sh")
echo "[3/4] Gen-UME forward folding                -> Job ${JOB_FF}"

JOB_CG=$(sbatch --parsable "${SCRIPT_DIR}/eval_gen_ume_protein_ligand_conditioned_generation.sh")
echo "[4/4] Gen-UME ligand-conditioned generation  -> Job ${JOB_CG}"

BASE="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks"

echo ""
echo "=============================================="
echo " Phase 1 jobs submitted. Monitor with:"
echo "   squeue -u \$USER"
echo "=============================================="
echo ""
echo "Output directories:"
echo "  LigandMPNN baseline: ${BASE}/ligandmpnn_baseline_${EVAL_TIMESTAMP}/"
echo "  Gen-UME inv. fold:   ${BASE}/gen_ume_pl_inverse_folding_${EVAL_TIMESTAMP}/"
echo "  Gen-UME fwd. fold:   ${BASE}/gen_ume_pl_forward_folding_${EVAL_TIMESTAMP}/"
echo "  Gen-UME cond. gen:   ${BASE}/gen_ume_pl_conditioned_gen_${EVAL_TIMESTAMP}/"
echo ""
echo "=============================================="
echo " Phase 2: After Phase 1 completes, submit co-folding batch jobs."
echo " Run each command below (adjust paths if needed):"
echo "=============================================="
echo ""

for label_csv in \
    "ligandmpnn_baseline:${BASE}/ligandmpnn_baseline_${EVAL_TIMESTAMP}/ligandmpnn_baseline_results.csv" \
    "gen_ume_inv_fold:${BASE}/gen_ume_pl_inverse_folding_${EVAL_TIMESTAMP}/inverse_folding_results.csv" \
    "gen_ume_fwd_fold:${BASE}/gen_ume_pl_forward_folding_${EVAL_TIMESTAMP}/forward_folding_results.csv" \
    "gen_ume_cond_gen:${BASE}/gen_ume_pl_conditioned_gen_${EVAL_TIMESTAMP}/conditioned_gen_results.csv"; do

    LABEL="${label_csv%%:*}"
    CSV="${label_csv#*:}"
    COFOLD_DIR="${BASE}/cofold_${LABEL}_${EVAL_TIMESTAMP}"

    cat <<CMD
# Co-fold for ${LABEL}:
cd /cv/home/lisanzas/lobster && uv run python -m lobster.cmdline.submit_cofold_batch \\
    --eval_csv "${CSV}" \\
    --output_dir "${COFOLD_DIR}" \\
    --backend protenix

CMD
done

echo "=============================================="
echo " Phase 3: After co-folding completes, merge results:"
echo "=============================================="
echo ""

for label_csv in \
    "ligandmpnn_baseline:${BASE}/ligandmpnn_baseline_${EVAL_TIMESTAMP}/ligandmpnn_baseline_results.csv" \
    "gen_ume_inv_fold:${BASE}/gen_ume_pl_inverse_folding_${EVAL_TIMESTAMP}/inverse_folding_results.csv" \
    "gen_ume_fwd_fold:${BASE}/gen_ume_pl_forward_folding_${EVAL_TIMESTAMP}/forward_folding_results.csv" \
    "gen_ume_cond_gen:${BASE}/gen_ume_pl_conditioned_gen_${EVAL_TIMESTAMP}/conditioned_gen_results.csv"; do

    LABEL="${label_csv%%:*}"
    CSV="${label_csv#*:}"
    COFOLD_DIR="${BASE}/cofold_${LABEL}_${EVAL_TIMESTAMP}"
    MERGED="${CSV%.csv}_with_cofold.csv"

    ID_COL="pdb_id"
    if [[ "${LABEL}" == "gen_ume_cond_gen" ]]; then
        ID_COL="ligand_id"
    fi

    cat <<CMD
# Merge for ${LABEL}:
cd /cv/home/lisanzas/lobster && uv run python -m lobster.cmdline.merge_cofold_results \\
    --results_dir "${COFOLD_DIR}/results" \\
    --eval_csv "${CSV}" \\
    --id_col "${ID_COL}" \\
    --output "${MERGED}" \\
    --parse_structures \\
    --data_dir "${DATA_DIR}"

CMD
done

echo "=============================================="
echo " Done. All Phase 1 SLURM jobs have been submitted."
echo "=============================================="
