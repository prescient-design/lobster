#!/bin/bash
#SBATCH --job-name=sair_pl_sm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-79  # 80 array jobs, each processing ~5K entries
#SBATCH --output=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/sair_pl_sm_%A_%a.out
#SBATCH --error=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/sair_pl_sm_%A_%a.err

# =============================================================================
# SAIR Protein-Ligand Dataset Processing - Small Resource Array Job
# =============================================================================
# Uses smaller resources for faster scheduling
# 80 jobs x ~5K entries each = 400K total
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1
mkdir -p slurm/logs

# Configuration
CSV_PATH="/data2/smdd/BindingAffinity/SAIR/sair/sair_balanced_split.csv"
OUTPUT_DIR="/data2/lisanzas/sair_protein_ligand"
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


