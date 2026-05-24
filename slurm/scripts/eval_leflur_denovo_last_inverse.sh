#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_inverse.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_inverse.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_denovo_inv
#SBATCH -t 1-00:00:00
#SBATCH -q llm

CKPT="${CKPT:-/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-08T17-09-23/last.ckpt}"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
EVAL_PREFIX="${EVAL_PREFIX:-gen_ume_denovo_last}"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/${EVAL_PREFIX}_ckpt_${EVAL_TIMESTAMP}_cameo_inverse_folding"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last

cd /cv/home/lisanzas/lobster
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name experiment/research/generate_inverse_folding_denovo_cameo \
    model.ckpt_path="${CKPT}" \
    output_dir="${OUT_DIR}"
