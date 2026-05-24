#!/bin/bash
# Full post-generation distillation pipeline.
#
# Run this after the 100-replica Proteina-Complexa generation+evaluation job completes.
# It executes the filter → convert → cluster → validate chain sequentially.
#
# Usage:
#   bash slurm/scripts/run_distillation_pipeline.sh

set -euo pipefail

YAML_CONFIG="/cv/scratch/u/lisanzas/proteina-complexa/configs/targets/ligand_targets_dict.yaml"
OUTPUT_DIR="/cv/scratch/u/lisanzas/distillation_passing"
PT_DIR="/cv/scratch/u/lisanzas/distillation_dataset/train"
PLINDER_DIR="/cv/scratch/u/lisanzas/plinder_processed/train"
PB_DIR="/cv/data/ai4dd/data2/lisanzas/pdb_bind_12_15_25/test"

cd /cv/home/lisanzas/lobster

echo "============================================"
echo "Step 1: Filter passing designs"
echo "============================================"
uv run python scripts/filter_proteina_results.py \
    --run_patterns "plinder_100rep_" "plinder_top30_" \
    --yaml_config "$YAML_CONFIG" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "Step 2: Convert to Gen-UME .pt format"
echo "============================================"
uv run python scripts/convert_proteina_to_genume.py \
    --input_csv "${OUTPUT_DIR}/passing_designs.csv" \
    --output_dir "$PT_DIR"

echo ""
echo "============================================"
echo "Step 3: Build ligand SMILES clusters"
echo "============================================"
uv run python scripts/build_ligand_clusters.py \
    --input_csv "${OUTPUT_DIR}/passing_designs.csv" \
    --pt_dir "$PT_DIR" \
    --output "/cv/scratch/u/lisanzas/distillation_dataset/ligand_clusters.pt"

echo ""
echo "============================================"
echo "Step 4: Validate datasets"
echo "============================================"
uv run python scripts/validate_distillation_dataset.py \
    --distillation_dir "$PT_DIR" \
    --plinder_dir "$PLINDER_DIR" \
    --posebusters_dir "$PB_DIR"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo "Passing designs:   ${OUTPUT_DIR}/passing_designs.csv"
echo "Cofold CSV:        ${OUTPUT_DIR}/cofold_inputs.csv"
echo "Training .pt dir:  ${PT_DIR}"
echo "Cluster file:      /cv/scratch/u/lisanzas/distillation_dataset/ligand_clusters.pt"
echo ""
echo "Next steps:"
echo "  1. Submit cofold validation:  bash slurm/scripts/submit_distillation_cofold.sh --submit"
echo "  2. Train with PLINDER config: data=structure_ligand_pdb_sair_plinder"
echo "  3. Train with distillation:   data=structure_ligand_distillation"
