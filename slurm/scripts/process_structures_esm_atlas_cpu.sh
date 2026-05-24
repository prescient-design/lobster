#!/bin/bash
#SBATCH --job-name=process_esm_atlas_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=240G  # cpu nodes have 256GB memory (leave headroom for system)
#SBATCH --time=48:00:00
#SBATCH --output=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/process_esm_atlas_cpu_%j.out
#SBATCH --error=/homefs/home/lisanzas/scratch/Develop/lobster/slurm/logs/process_esm_atlas_cpu_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@gene.com

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Navigate to the project directory
cd /homefs/home/lisanzas/scratch/Develop/lobster || exit 1

# Create logs directory if it doesn't exist
mkdir -p slurm/logs

# Print environment info
echo "Python version:"
uv run python --version

echo "Starting structure processing..."
echo "Input:  /data2/ume/simplefold_dataset/esm_atlas/"
echo "Output: /data2/ume/simplefold_dataset/train_processed/"
echo "Workers: 128"
echo "=========================================="

# Run the processing script
uv run python scripts/process_structures.py \
    --input-dir /data2/ume/simplefold_dataset/esm_atlas/ \
    --output-dir /data2/ume/simplefold_dataset/train_processed/ \
    --num-workers 128

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $?"
echo "=========================================="

# Print summary if available
if [ -f /data2/ume/simplefold_dataset/train_processed/failed_files.txt ]; then
    echo "Failed files found. Check: /data2/ume/simplefold_dataset/train_processed/failed_files.txt"
    echo "Number of failed files: $(wc -l < /data2/ume/simplefold_dataset/train_processed/failed_files.txt)"
fi

