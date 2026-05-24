#!/bin/bash
#SBATCH --job-name=pdbbind_bm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=/data2/lisanzas/slurm_logs/pdbbind_bm_%A_%a.out
#SBATCH --error=/data2/lisanzas/slurm_logs/pdbbind_bm_%A_%a.err

# =============================================================================
# PDBBind Processing - Update existing .pt files with SMILES and bond_matrix
# =============================================================================
# 20 jobs x ~1.4K files each = ~27K ligand files total
# Input: /data2/lisanzas/pdb_bind/ (existing)
# Output: Updates in place (or use --output-dir for new location)
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1
mkdir -p /data2/lisanzas/slurm_logs

# Configuration
INPUT_DIR="/data2/lisanzas/pdb_bind"
OUTPUT_DIR="/data2/lisanzas/pdb_bind_12_15_25"
# Total ligand files: train=21835 + val=2729 + test=2730 = 27294
TOTAL_FILES=27294
NUM_JOBS=20
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

# Use update mode - reads existing .pt files and adds smiles/bond_matrix
# Files are copied to output dir with the new fields
uv run python scripts/process_pdbbind_protein_ligand.py \
    --mode update \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --num-workers $NUM_WORKERS

echo "=========================================="
echo "Completed at: $(date), Exit: $?"
echo "=========================================="

