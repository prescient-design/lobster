#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_competitor/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_competitor/%J_%x.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_competitor
#SBATCH -t 4:00:00
#SBATCH -q llm

# Evaluate competitor co-design models (LaProteina, DPLM2) for benchmarking.
# Submit: sbatch --export=MODEL=laproteina slurm/scripts/eval_competitor_unconditional.sh

set -eo pipefail
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_competitor

MODEL="${MODEL:?Must set MODEL (laproteina or dplm2)}"
OUTPUT_DIR="/cv/scratch/u/lisanzas/evaluations/benchmark_${MODEL}_unconditional"

cd /cv/home/lisanzas/lobster

echo "=== Evaluating ${MODEL} at $(date) ==="
echo "Output: ${OUTPUT_DIR}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python scripts/eval_competitor_unconditional.py \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}"

echo "=== Done at $(date) ==="
