#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_ss_balanced_fwd_mf.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_ss_balanced_fwd_mf.err
#SBATCH --mem=128G
#SBATCH --job-name=ss_bal_fwd_mf
#SBATCH -t 1-00:00:00
#SBATCH -q llm

CKPT="/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_ted_cath_ss_balanced_2026-03-14T15-41-36_2026-03-18T12-20-59.ckpt"
EVAL_TIMESTAMP="2026-03-18T12-20-59"
EVAL_PREFIX="gen_ume_denovo_ted_cath_ss_balanced"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/${EVAL_PREFIX}_ckpt_${EVAL_TIMESTAMP}_multiflow_forward_folding"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last

cd /cv/home/lisanzas/lobster

uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name experiment/research/generate_forward_folding_denovo_multiflow \
    model.ckpt_path="${CKPT}" \
    output_dir="${OUT_DIR}"
