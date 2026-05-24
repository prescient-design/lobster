#!/bin/bash
#SBATCH --job-name=geom_test
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --output=/data2/lisanzas/slurm_logs/geom_test_%A_%a.out
#SBATCH --error=/data2/lisanzas/slurm_logs/geom_test_%A_%a.err

# =============================================================================
# GEOM Test Processing - Add bond_matrix from S3 SDF files
# =============================================================================
# 10 jobs x ~3.1K files each = ~31K files total
# Input: /data/bucket/lisanza/structures/GEOM/processed/test/
# Output: /data2/lisanzas/geom_12_15_25/test/
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1
mkdir -p /data2/lisanzas/slurm_logs

# Configuration
INPUT_DIR="/data/bucket/lisanza/structures/GEOM/processed/test"
OUTPUT_DIR="/data2/lisanzas/geom_12_15_25/test"
TOTAL_FILES=30936
NUM_JOBS=10
FILES_PER_JOB=$((TOTAL_FILES / NUM_JOBS + 1))
NUM_WORKERS=16

START_IDX=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * FILES_PER_JOB))

if [ $END_IDX -gt $TOTAL_FILES ]; then
    END_IDX=$TOTAL_FILES
fi

echo "Processing files: $START_IDX to $END_IDX ($(($END_IDX - $START_IDX)) files)"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

uv run python scripts/process_geom_ligand.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --num-workers $NUM_WORKERS

echo "=========================================="
echo "Completed at: $(date), Exit: $?"
echo "=========================================="





