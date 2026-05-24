#!/bin/bash
#SBATCH --job-name=process_sair_pl
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-39  # 40 array jobs, each processing ~10K entries
#SBATCH --output=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/process_sair_pl_%A_%a.out
#SBATCH --error=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/process_sair_pl_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@gene.com

# =============================================================================
# SAIR Protein-Ligand Dataset Processing - Array Job
# =============================================================================
# This script processes the SAIR protein-ligand dataset in parallel using
# SLURM array jobs. Each array task processes a subset of the 400K entries.
#
# Dataset: /data2/smdd/BindingAffinity/SAIR/sair/sair_balanced_split.csv
# Total entries: 400,508
# - Train: 279,980
# - Test: 81,780
# - Val: 38,748
#
# With 40 array jobs, each processes ~10,000 entries
# =============================================================================

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Navigate to the project directory
cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1

# Create logs directory if it doesn't exist
mkdir -p slurm/logs

# Configuration
CSV_PATH="/data2/smdd/BindingAffinity/SAIR/sair/sair_balanced_split.csv"
OUTPUT_DIR="/data2/lisanzas/sair_protein_ligand"
TOTAL_ENTRIES=400508
NUM_JOBS=40
ENTRIES_PER_JOB=$((TOTAL_ENTRIES / NUM_JOBS + 1))  # Round up
NUM_WORKERS=64  # Match cpus-per-task

# Calculate start and end indices for this array task
START_IDX=$((SLURM_ARRAY_TASK_ID * ENTRIES_PER_JOB))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * ENTRIES_PER_JOB))

# Make sure we don't exceed total entries
if [ $END_IDX -gt $TOTAL_ENTRIES ]; then
    END_IDX=$TOTAL_ENTRIES
fi

echo "Processing entries: $START_IDX to $END_IDX ($(($END_IDX - $START_IDX)) entries)"
echo "Using $NUM_WORKERS workers"
echo "=========================================="

# Run the processing script
uv run python scripts/process_sair_protein_ligand.py \
    --csv-path "$CSV_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --num-workers $NUM_WORKERS

EXIT_CODE=$?

# Print completion information
echo "=========================================="
echo "Array task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

