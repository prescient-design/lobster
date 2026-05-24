#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_forward.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_forward.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_denovo_fwd
#SBATCH -t 1-00:00:00
#SBATCH -q llm

CKPT="${CKPT:-/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-08T17-09-23/last.ckpt}"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
EVAL_PREFIX="${EVAL_PREFIX:-gen_ume_denovo_last}"
NSTEPS_SUFFIX=""
[ -n "${NSTEPS}" ] && NSTEPS_SUFFIX="_nsteps${NSTEPS}"
TEMP_STRUC_SUFFIX=""
[ -n "${TEMPERATURE_STRUC}" ] && TEMP_STRUC_SUFFIX="_temp_struc${TEMPERATURE_STRUC}"
STOCH_STRUC_SUFFIX=""
[ -n "${STOCHASTICITY_STRUC}" ] && STOCH_STRUC_SUFFIX="_stoch_struc${STOCHASTICITY_STRUC}"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/${EVAL_PREFIX}_ckpt_${EVAL_TIMESTAMP}${NSTEPS_SUFFIX}${TEMP_STRUC_SUFFIX}${STOCH_STRUC_SUFFIX}_cameo_forward_folding"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last

cd /cv/home/lisanzas/lobster
EXTRA_ARGS=()
[ -n "${SEED}" ] && EXTRA_ARGS+=(seed="${SEED}")
[ -n "${NSTEPS}" ] && EXTRA_ARGS+=(generation.nsteps="${NSTEPS}")
[ -n "${TEMPERATURE_STRUC}" ] && EXTRA_ARGS+=(generation.temperature_struc="${TEMPERATURE_STRUC}")
[ -n "${STOCHASTICITY_STRUC}" ] && EXTRA_ARGS+=(generation.stochasticity_struc="${STOCHASTICITY_STRUC}")

uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name experiment/research/generate_forward_folding_denovo_cameo \
    model.ckpt_path="${CKPT}" \
    output_dir="${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"
