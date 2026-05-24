#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_uncond.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last/%J_uncond.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_denovo_uncond
#SBATCH -t 1-00:00:00
#SBATCH -q llm

CKPT="${CKPT:-/cv/scratch/u/lisanzas/gen_ume_denovo/runs/2026-03-08T17-09-23/last.ckpt}"
EVAL_TIMESTAMP="${EVAL_TIMESTAMP:-$(date +%Y-%m-%dT%H-%M-%S)}"
EVAL_PREFIX="${EVAL_PREFIX:-gen_ume_denovo_last}"
CONFIG_NAME="${CONFIG_NAME:-generate_unconditional_denovo}"
STOCH_SUFFIX=""
[ -n "${STOCHASTICITY_SEQ}" ] && [ -n "${STOCHASTICITY_STRUC}" ] && STOCH_SUFFIX="_seq${STOCHASTICITY_SEQ}_struc${STOCHASTICITY_STRUC}"
TEMP_SUFFIX=""
[ -n "${TEMPERATURE_SEQ}" ] && [ -n "${TEMPERATURE_STRUC}" ] && TEMP_SUFFIX="_tseq${TEMPERATURE_SEQ}_tstruc${TEMPERATURE_STRUC}"
BIAS_SUFFIX="${BIAS_SUFFIX:-}"
UNCOND_DIR="/cv/scratch/u/lisanzas/evaluations/${EVAL_PREFIX}_ckpt_${EVAL_TIMESTAMP}_unconditional${STOCH_SUFFIX}${TEMP_SUFFIX}${BIAS_SUFFIX}"
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_denovo_last

cd /cv/home/lisanzas/lobster
EXTRA_ARGS=()
[ -n "${STOCHASTICITY_SEQ}" ] && EXTRA_ARGS+=(generation.stochasticity_seq="${STOCHASTICITY_SEQ}")
[ -n "${STOCHASTICITY_STRUC}" ] && EXTRA_ARGS+=(generation.stochasticity_struc="${STOCHASTICITY_STRUC}")
[ -n "${TEMPERATURE_SEQ}" ] && EXTRA_ARGS+=(generation.temperature_seq="${TEMPERATURE_SEQ}")
[ -n "${TEMPERATURE_STRUC}" ] && EXTRA_ARGS+=(generation.temperature_struc="${TEMPERATURE_STRUC}")
[ -n "${LOGIT_BIAS_OVERRIDE}" ] && EXTRA_ARGS+=($LOGIT_BIAS_OVERRIDE)
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name "${CONFIG_NAME}" \
    model.ckpt_path="${CKPT}" \
    output_dir="${UNCOND_DIR}" \
    "${EXTRA_ARGS[@]}"

# Secondary structure analysis
if [ -d "${UNCOND_DIR}" ]; then
    uv run python scripts/analyze_secondary_structure.py \
        --uncond-only \
        --uncond-dir "${UNCOND_DIR}" \
        --output "${UNCOND_DIR}/uncond_sse_index.parquet"
fi

# TM-score novelty analysis (denovo vs PDB)
PDB_REPS_PDB="/cv/scratch/u/lisanzas/pdb_seqid40_cluster_reps_pdb"
if [ -d "${UNCOND_DIR}" ]; then
    uv run python scripts/analyze_tm_score_novelty.py \
        --uncond-dir "${UNCOND_DIR}" \
        --denovo-reps-dir "/cv/scratch/u/lisanzas/denovo_dataset/clustered" \
        --pdb-reps-pdb-dir "${PDB_REPS_PDB}" \
        --foldseek-bin "/cv/home/lisanzas/lobster/src/lobster/metrics/foldseek/bin" \
        --use-existing-clusters
fi
