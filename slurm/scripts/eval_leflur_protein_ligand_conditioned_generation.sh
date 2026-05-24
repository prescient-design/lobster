#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_protein_ligand_conditioned_gen/%J.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_protein_ligand_conditioned_gen/%J.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_pl_cg
#SBATCH -t 1-00:00:00
#SBATCH -q llm

# Gen-UME ligand-conditioned protein generation on 17 diverse PoseBusters ligands
# ESMFold self-consistency validation (scTM, scRMSD, pLDDT)
# Co-folding validation done separately via SLURM batch (submit_cofold_batch.py)
# 10 designs per ligand at length 100 = 170 sequences total

CKPT="${CKPT:-/cv/scratch/u/lisanzas/gen_ume_protein_ligand_no_geom_medium/runs//2026-03-11T13-22-20/last.ckpt}"
DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/processed/posebusters_benchmark_no_overlap/"
RAW_DATA_DIR="/cv/home/lisanzas/lobster/data/posebusters/posebusters_benchmark_set/"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
EVAL_PREFIX="${EVAL_PREFIX:-gen_ume_pl_conditioned_gen}"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/${EVAL_PREFIX}_${EVAL_TIMESTAMP}"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_protein_ligand_conditioned_gen
mkdir -p "${OUT_DIR}"

cd /cv/home/lisanzas/lobster

uv run python -m lobster.cmdline.evaluate_ligand_conditioned_protein_generation \
    --checkpoint "${CKPT}" \
    --data_dir "${DATA_DIR}" \
    --raw_data_dir "${RAW_DATA_DIR}" \
    --output "conditioned_gen_results.csv" \
    --structure_path "${OUT_DIR}" \
    --length 100 \
    --num_designs 10 \
    --nsteps 100 \
    --save_structures \
    --num_samples 17

echo "Results saved to ${OUT_DIR}"
