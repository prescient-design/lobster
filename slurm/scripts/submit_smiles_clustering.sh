#!/bin/bash
# Submit all 5 SMILES clustering jobs in parallel
# Small datasets (distillation, redesign): ~2 min each
# Medium (PDBBind): ~10 min
# Large (PLINDER): ~20 min
# Largest (SAIR): ~30 min with 48 workers

set -euo pipefail

LOBSTER_DIR="/cv/home/lisanzas/lobster"
LOG_DIR="/cv/scratch/u/lisanzas/slurm_logs"
mkdir -p "$LOG_DIR"

for dataset in distillation redesign pdbbind plinder sair; do
    case $dataset in
        sair)    cpus=48; mem="128G"; time="01:00:00" ;;
        plinder) cpus=32; mem="64G";  time="01:00:00" ;;
        pdbbind) cpus=16; mem="32G";  time="00:30:00" ;;
        *)       cpus=8;  mem="16G";  time="00:10:00" ;;
    esac

    sbatch --partition=himem --account=llm --qos=llm \
        --nodes=1 --ntasks-per-node=1 \
        --cpus-per-task=$cpus --mem=$mem -t $time \
        --job-name="sclust-${dataset}" \
        -o "${LOG_DIR}/smiles_cluster_${dataset}_%j.out" \
        -e "${LOG_DIR}/smiles_cluster_${dataset}_%j.err" \
        --wrap="cd ${LOBSTER_DIR} && uv run python scripts/build_canonical_smiles_clusters.py --dataset ${dataset} --workers ${cpus}"

    echo "Submitted: ${dataset} (${cpus} CPUs, ${mem})"
done
