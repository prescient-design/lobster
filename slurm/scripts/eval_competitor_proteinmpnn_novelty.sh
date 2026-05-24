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
#SBATCH --job-name=eval_competitor_pmpnn_novelty
#SBATCH -t 2:00:00
#SBATCH -q llm

# Novelty-only pass for Proteina/Genie2 + ProteinMPNN benchmark.
# Reuses already-written ESMFold PDBs, Foldseek clusters, and SSE.
#
# Submit: sbatch --export=MODEL=proteina slurm/scripts/eval_competitor_proteinmpnn_novelty.sh
#         sbatch --export=MODEL=genie2   slurm/scripts/eval_competitor_proteinmpnn_novelty.sh

set -eo pipefail
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_competitor

MODEL="${MODEL:?Must set MODEL (proteina or genie2)}"
OUTPUT_DIR="/cv/scratch/u/lisanzas/evaluations/benchmark_${MODEL}_pmpnn_unconditional"

cd /cv/home/lisanzas/lobster

echo "=== Novelty for ${MODEL} at $(date) ==="
echo "Output: ${OUTPUT_DIR}"

uv run python scripts/eval_competitor_proteinmpnn.py \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    --skip-esmfold \
    --skip-clustering \
    --skip-sse

echo "=== Done at $(date) ==="
