#!/bin/bash
#SBATCH --job-name=geom_bm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-99
#SBATCH --output=/data2/lisanzas/slurm_logs/geom_bm_%A_%a.out
#SBATCH --error=/data2/lisanzas/slurm_logs/geom_bm_%A_%a.err

# =============================================================================
# GEOM Processing - Add bond_matrix from S3 SDF files
# =============================================================================
# 100 jobs x ~2.5K files each = ~247K files total
# Input: /data/bucket/lisanza/structures/GEOM/processed/train/
# Output: /data2/lisanzas/geom_12_15_25/train/
# Note: Reads SDF files from S3 - may be slower than local processing
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1
mkdir -p /data2/lisanzas/slurm_logs

# Configuration
INPUT_DIR="/data/bucket/lisanza/structures/GEOM/processed/train"
OUTPUT_DIR="/data2/lisanzas/geom_12_15_25/train"
TOTAL_FILES=246840
NUM_JOBS=100
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

