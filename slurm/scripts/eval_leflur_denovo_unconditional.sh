#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_uncond/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_uncond/%J_%x.err
#SBATCH --mem=128G
#SBATCH --job-name=eval_uncond
#SBATCH -t 1-00:00:00
#SBATCH -q llm

# Gen-UME Unconditional Generation + SS Analysis
# General script for any unconditional Gen-UME run.
#
# Usage:
#   sbatch slurm/scripts/eval_gen_ume_denovo_unconditional.sh
#   sbatch slurm/scripts/eval_gen_ume_denovo_unconditional.sh generate_unconditional_750M_L100
#   CONFIG=generate_unconditional_450M OUTPUT_DIR=/custom/path sbatch ...
#
# Args: CONFIG_NAME (optional, default: generate_unconditional_denovo)
# Env:  CONFIG, OUTPUT_DIR (optional overrides)

CONFIG="${CONFIG:-${1:-generate_unconditional_denovo}}"
CONFIG_PATH="src/lobster/hydra_config/experiment"
HYDRA_OVERRIDES="${HYDRA_OVERRIDES:-}"

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/eval_gen_ume_uncond

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "Config: ${CONFIG}"

# Resolve output_dir: use OUTPUT_DIR if set, else read from config yaml
if [ -n "${OUTPUT_DIR}" ]; then
    UNCOND_DIR="${OUTPUT_DIR}"
else
    UNCOND_DIR=$(uv run python -c "
import yaml
from pathlib import Path
p = Path('${CONFIG_PATH}/${CONFIG}.yaml')
cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
d = cfg.get('output_dir', '')
if d and not Path(d).is_absolute():
    d = str(Path('.').resolve() / d)
print(d or '')
")
fi

echo "Output dir: ${UNCOND_DIR}"

# 1. Unconditional generation
uv run python -m lobster.cmdline.generate \
    --config-path "../hydra_config/experiment" \
    --config-name "${CONFIG}" \
    $HYDRA_OVERRIDES

# 2. Secondary structure analysis with per-length readout
if [ -n "${UNCOND_DIR}" ] && [ -d "${UNCOND_DIR}" ]; then
    echo ""
    echo "=== Secondary Structure Analysis (per length) ==="
    uv run python scripts/analyze_secondary_structure.py \
        --uncond-only \
        --uncond-dir "${UNCOND_DIR}" \
        --output "${UNCOND_DIR}/uncond_sse_index.parquet"
else
    echo "Unconditional output dir not found or empty: ${UNCOND_DIR:-<unset>}"
fi
