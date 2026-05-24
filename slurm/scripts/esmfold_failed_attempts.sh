#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 8
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/esmfold_failed_attempts/%J.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/esmfold_failed_attempts/%J.err
#SBATCH --mem=64G
#SBATCH --job-name=esmfold_failed
#SBATCH -t 0-04:00:00
#SBATCH -q llm

# ESMFold each saved forward-fold-rejected sequence against the saved initial
# backbone, to enable building the SR-QC vs ESMFold-QC concordance matrix.
#
# Required env var: CONC_DIR  (concordance run output directory)

set -euo pipefail

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/esmfold_failed_attempts

cd /cv/home/lisanzas/lobster

if [ -z "${CONC_DIR:-}" ]; then
    echo "ERROR: CONC_DIR must be set" >&2
    exit 1
fi

echo "Concordance dir: ${CONC_DIR}"

uv run python scripts/esmfold_failed_attempts.py \
    --concordance-dir "${CONC_DIR}"
