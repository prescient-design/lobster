#!/usr/bin/env bash
# Inverse folding on MultiFlow test set using gen_ume_denovo_last checkpoint (TM 0.667 CAMEO eval)
# Checkpoint: 2026-03-11T12-11-53 snapshot
# Usage: sbatch slurm/scripts/eval_gen_ume_denovo_last_inverse_multiflow.sh

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_inverse_multiflow.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_inverse_multiflow.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_denovo_inv_mf
#SBATCH -t 1-00:00:00
#SBATCH -q llm

CKPT="${CKPT:-/cv/scratch/u/lisanzas/evaluations/checkpoint_snapshots/gen_ume_denovo_last_2026-03-08T17-09-23_2026-03-11T12-11-53.ckpt}"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/gen_ume_denovo_last_ckpt_2026-03-11T12-11-53_multiflow_inverse_folding"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last

cd /cv/home/lisanzas/lobster
EXTRA_ARGS=()
[ -n "${SEED}" ] && EXTRA_ARGS+=(seed="${SEED}")

uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name experiment/research/generate_inverse_folding_denovo_multiflow \
    model.ckpt_path="${CKPT}" \
    output_dir="${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"
