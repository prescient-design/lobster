#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_ligandmpnn_baseline/%J.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_ligandmpnn_baseline/%J.err
#SBATCH --mem=64G
#SBATCH --job-name=eval_lmpnn
#SBATCH -t 1-00:00:00
#SBATCH -q llm

# LigandMPNN inverse folding baseline on PoseBusters
# Runs LigandMPNN locally (GPU required)
# Co-folding validation done separately via SLURM batch (submit_cofold_batch.py)

DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap/"
RAW_DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/posebusters_benchmark_set/"
LIGANDMPNN_PATH="/cv/home/lisanzas/LigandMPNN"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/ligandmpnn_baseline_${EVAL_TIMESTAMP}"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_ligandmpnn_baseline
mkdir -p "${OUT_DIR}"

cd /cv/home/lisanzas/lobster

uv run python -m lobster.cmdline.evaluate_ligandmpnn_baseline \
    --data_dir "${DATA_DIR}" \
    --raw_data_dir "${RAW_DATA_DIR}" \
    --output "ligandmpnn_baseline_results.csv" \
    --structure_path "${OUT_DIR}" \
    --num_designs 10 \
    --temperature 0.1 \
    --use_local_ligandmpnn \
    --ligandmpnn_path "${LIGANDMPNN_PATH}" \
    --num_samples -1

echo "Results saved to ${OUT_DIR}"
