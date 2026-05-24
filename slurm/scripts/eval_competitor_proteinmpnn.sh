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
#SBATCH --job-name=eval_competitor_pmpnn
#SBATCH -t 4:00:00
#SBATCH -q llm

# Evaluate backbone-only competitor models (Proteina, Genie2) paired with
# ProteinMPNN-CA sequences. Mirrors the LaProteina/DPLM2 benchmark format.
#
# Submit: sbatch --export=MODEL=proteina slurm/scripts/eval_competitor_proteinmpnn.sh
#         sbatch --export=MODEL=genie2   slurm/scripts/eval_competitor_proteinmpnn.sh

set -eo pipefail
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_competitor

MODEL="${MODEL:?Must set MODEL (proteina or genie2)}"
OUTPUT_DIR="/cv/scratch/u/lisanzas/evaluations/benchmark_${MODEL}_pmpnn_unconditional"

cd /cv/home/lisanzas/lobster

echo "=== Evaluating ${MODEL} + ProteinMPNN-CA at $(date) ==="
echo "Output: ${OUTPUT_DIR}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python scripts/eval_competitor_proteinmpnn.py \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}"

echo "=== Done at $(date) ==="
