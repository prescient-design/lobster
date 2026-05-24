#!/bin/bash
#SBATCH --job-name=sair_bm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-79
#SBATCH --output=/data2/lisanzas/slurm_logs/sair_bm_%A_%a.out
#SBATCH --error=/data2/lisanzas/slurm_logs/sair_bm_%A_%a.err

# =============================================================================
# SAIR Processing - Add SMILES and bond_matrix to NEW output directory
# =============================================================================
# 80 jobs x ~5K entries each = 400K total
# Output: /data2/lisanzas/sair_12_15_25/
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1
mkdir -p /data2/lisanzas/slurm_logs

# Configuration
CSV_PATH="/data2/smdd/BindingAffinity/SAIR/sair/sair_balanced_split.csv"
OUTPUT_DIR="/data2/lisanzas/sair_12_15_25"
TOTAL_ENTRIES=400508
NUM_JOBS=80
ENTRIES_PER_JOB=$((TOTAL_ENTRIES / NUM_JOBS + 1))
NUM_WORKERS=16

START_IDX=$((SLURM_ARRAY_TASK_ID * ENTRIES_PER_JOB))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * ENTRIES_PER_JOB))

if [ $END_IDX -gt $TOTAL_ENTRIES ]; then
    END_IDX=$TOTAL_ENTRIES
fi

echo "Processing entries: $START_IDX to $END_IDX ($(($END_IDX - $START_IDX)) entries)"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

uv run python scripts/process_sair_protein_ligand.py \
    --csv-path "$CSV_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --num-workers $NUM_WORKERS

echo "=========================================="
echo "Completed at: $(date), Exit: $?"
echo "=========================================="
