#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_protein_ligand_inverse_folding/%J.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_protein_ligand_inverse_folding/%J.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_pl_if
#SBATCH -t 1-00:00:00
#SBATCH -q llm

# Gen-UME protein-ligand inverse folding evaluation on PoseBusters
# Co-folding validation done separately via SLURM batch (submit_cofold_batch.py)

CKPT="${CKPT:-/cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_medium/runs//2026-03-11T13-22-20/last.ckpt}"
DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap/"
RAW_DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/posebusters_benchmark_set/"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
EVAL_PREFIX="${EVAL_PREFIX:-gen_ume_pl_inverse_folding}"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/${EVAL_PREFIX}_${EVAL_TIMESTAMP}"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_protein_ligand_inverse_folding
mkdir -p "${OUT_DIR}"

cd /cv/home/lisanzas/lobster

uv run python -m lobster.cmdline.evaluate_protein_ligand_inverse_folding \
    --checkpoint "${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --raw_data_dir "${RAW_DATA_DIR}" \
    --output "inverse_folding_results.csv" \
    --structure_path "${OUT_DIR}" \
    --nsteps 100 \
    --decode_structure \
    --save_gt_structure \
    --num_samples -1

echo "Results saved to ${OUT_DIR}"
