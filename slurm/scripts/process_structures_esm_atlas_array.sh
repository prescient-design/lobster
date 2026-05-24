#!/bin/bash
#SBATCH --job-name=process_esm_atlas_array
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --array=0-9  # 10 array jobs, each processing 100 directories
#SBATCH --output=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/process_esm_atlas_array_%A_%a.out
#SBATCH --error=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/process_esm_atlas_array_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@gene.com

# NOTE: Using cpu partition (30 nodes available with 128 vCPU / 256GB mem each)
# If you want to use himem partition (3 nodes with 128 vCPU / 1TB mem), 
# change partition to himem, set --array=0-2, and update NUM_JOBS=3 below

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

# Calculate which subdirectories this array task should process
# ESM Atlas has directories named 000, 001, 002, ..., 999
# With 10 array jobs, each job processes 100 directories

INPUT_DIR="/data2/ume/simplefold_dataset/esm_atlas"
OUTPUT_DIR="/data2/ume/simplefold_dataset/train_processed"
TOTAL_DIRS=1000  # 000 to 999
NUM_JOBS=10      # Number of array tasks
DIRS_PER_JOB=$((TOTAL_DIRS / NUM_JOBS))

START_DIR=$((SLURM_ARRAY_TASK_ID * DIRS_PER_JOB))
END_DIR=$(((SLURM_ARRAY_TASK_ID + 1) * DIRS_PER_JOB - 1))

echo "This task will process directories: $(printf '%03d' $START_DIR) to $(printf '%03d' $END_DIR)"
echo "=========================================="

# Process each directory in this task's range
for ((dir_num=START_DIR; dir_num<=END_DIR; dir_num++)); do
    dir_name=$(printf '%03d' $dir_num)
    dir_path="${INPUT_DIR}/${dir_name}"
    
    if [ ! -d "$dir_path" ]; then
        echo "Directory $dir_path does not exist, skipping..."
        continue
    fi
    
    echo "Processing directory: $dir_name"
    
    # Run the processing script on this specific subdirectory
    uv run python scripts/process_structures.py \
        --input-dir "${dir_path}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-workers 128
    
    echo "Completed directory: $dir_name"
done

# Print completion information
echo "=========================================="
echo "Array task $SLURM_ARRAY_TASK_ID completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="

