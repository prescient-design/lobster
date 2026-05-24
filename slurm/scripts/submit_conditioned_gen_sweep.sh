#!/bin/bash
# Submit conditioned generation hyperparameter sweep — one GPU per config
# All run in parallel for maximum speed

set -euo pipefail

LOBSTER_DIR="/cv/home/lisanzas/lobster"
CKPT="${CKPT:-/cv/scratch/u/lisanzas/evaluations/protein_ligand_benchmarks/checkpoints_all_latest_0404/last.ckpt}"
LOG_DIR="/cv/scratch/u/lisanzas/slurm_logs/cg_sweep"
OUT_DIR="/cv/scratch/u/lisanzas/evaluations/conditioned_gen_sweep"
NUM_SAMPLES="${NUM_SAMPLES:-30}"
NUM_DESIGNS="${NUM_DESIGNS:-5}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

CONFIGS="baseline low_lig_temp very_low_lig_temp low_all_temp more_steps many_steps power_lig_sched log_lig_sched low_temp_more_steps low_temp_power_sched deterministic high_lig_temp"

for config in $CONFIGS; do
    sbatch --partition=ai4dd-b200 --account=llm --qos=llm \
        --nodes=1 --ntasks-per-node=1 --gres=gpu:b200:1 --cpus-per-task=16 --mem=128G \
        -t 04:00:00 --job-name="cg-sw-${config}" \
        -o "${LOG_DIR}/${config}_%j.out" \
        -e "${LOG_DIR}/${config}_%j.err" \
        --wrap="
cd ${LOBSTER_DIR}
uv run python scripts/conditioned_gen_sweep.py \
    --checkpoint '${CKPT}' \
    --config ${config} \
    --num_samples ${NUM_SAMPLES} \
    --num_designs ${NUM_DESIGNS} \
    --output_dir '${OUT_DIR}'
echo 'DONE: ${config}'
"
    echo "Submitted: ${config}"
done

echo ""
echo "All ${#CONFIGS} configs submitted. Results will be in ${OUT_DIR}/"
echo "Monitor: squeue -u \$USER | grep cg-sw"
