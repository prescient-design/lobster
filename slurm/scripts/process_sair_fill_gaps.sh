#!/bin/bash
#SBATCH --job-name=sair_fill
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-34
#SBATCH --output=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/sair_fill_%A_%a.out
#SBATCH --error=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/sair_fill_%A_%a.err

# =============================================================================
# Fill in missing gaps from SAIR processing (2659 entries across 35 gaps)
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID (Array Task: $SLURM_ARRAY_TASK_ID)"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "=========================================="

cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1

CSV_PATH="/data2/smdd/BindingAffinity/SAIR/sair/sair_balanced_split.csv"
OUTPUT_DIR="/data2/lisanzas/sair_protein_ligand"

# Define all gap ranges (start, end_exclusive)
GAPS=(
    "946,947"
    "100447,100448"
    "100940,100941"
    "101336,101337"
    "102103,102104"
    "109856,110154"
    "127362,127363"
    "127371,127372"
    "127551,127552"
    "127575,127576"
    "130176,130182"
    "159137,159138"
    "185191,185259"
    "207208,207209"
    "234826,235329"
    "240927,240928"
    "240930,240931"
    "265053,265055"
    "265056,265371"
    "267536,267537"
    "320099,320448"
    "324900,325455"
    "336090,336091"
    "336094,336095"
    "336101,336102"
    "336120,336121"
    "336164,336165"
    "336203,336204"
    "336216,336217"
    "336239,336240"
    "359559,359560"
    "359656,359657"
    "359664,359665"
    "380261,380532"
    "385272,385539"
)

# Get this job's gap range
GAP_INFO=${GAPS[$SLURM_ARRAY_TASK_ID]}
START_IDX=$(echo $GAP_INFO | cut -d',' -f1)
END_IDX=$(echo $GAP_INFO | cut -d',' -f2)

echo "Processing gap $SLURM_ARRAY_TASK_ID: entries $START_IDX to $END_IDX ($(($END_IDX - $START_IDX)) entries)"
echo "=========================================="

uv run python scripts/process_sair_protein_ligand.py \
    --csv-path "$CSV_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    --num-workers 16

echo "=========================================="
echo "Completed at: $(date), Exit: $?"
echo "=========================================="
