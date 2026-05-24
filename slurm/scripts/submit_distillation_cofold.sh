#!/bin/bash
# Submit Boltz2 and Protenix co-folding for passing Proteina-Complexa designs.
#
# Usage:
#   bash slurm/scripts/submit_distillation_cofold.sh [--submit]
#
# This prepares input JSONs from the filtered passing_designs.csv and creates
# SLURM array jobs for both backends. Pass --submit to actually submit.

set -euo pipefail

PASSING_CSV="/cv/home/lisanzas/distillation_passing/cofold_inputs.csv"
COFOLD_BASE="/cv/scratch/u/lisanzas/distillation_cofold"

if [ ! -f "$PASSING_CSV" ]; then
    echo "Error: $PASSING_CSV not found. Run filter_proteina_results.py first."
    exit 1
fi

SUBMIT_FLAG=""
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT_FLAG="--submit"
fi

cd /cv/home/lisanzas/lobster

echo "=== Preparing Protenix co-fold ==="
uv run python -m lobster.cmdline.submit_cofold_batch \
    --eval_csv "$PASSING_CSV" \
    --output_dir "${COFOLD_BASE}/protenix" \
    --backend protenix \
    --id_col pdb_id \
    --sequence_col sequence \
    --smiles_col smiles \
    --time_limit "1:00:00" \
    --mem "64G" \
    $SUBMIT_FLAG

echo ""
echo "=== Preparing Boltz2 co-fold ==="
uv run python -m lobster.cmdline.submit_cofold_batch \
    --eval_csv "$PASSING_CSV" \
    --output_dir "${COFOLD_BASE}/boltz" \
    --backend boltz \
    --id_col pdb_id \
    --sequence_col sequence \
    --smiles_col smiles \
    --time_limit "1:00:00" \
    --mem "64G" \
    $SUBMIT_FLAG

echo ""
echo "Done. When jobs complete, merge results with:"
echo "  uv run python -m lobster.cmdline.merge_cofold_results --results_dir ${COFOLD_BASE}/protenix/results --output_csv ${COFOLD_BASE}/protenix_results.csv"
echo "  uv run python -m lobster.cmdline.merge_cofold_results --results_dir ${COFOLD_BASE}/boltz/results --output_csv ${COFOLD_BASE}/boltz_results.csv"
