#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/test_%A.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/test_%A.err
#SBATCH --job-name=proteina-test
#SBATCH -t 01:00:00

set -euo pipefail

PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
TASK_LIST="${PROTEINA_DIR}/configs/targets/plinder_task_names.txt"
OUTPUT_BASE="/cv/scratch/u/lisanzas/proteina_gen_output"

TASK_NAME=$(head -1 "$TASK_LIST")
echo "=== Testing with task: $TASK_NAME ==="

cd "$PROTEINA_DIR"
source .venv/bin/activate
source env.sh

OUTDIR="${OUTPUT_BASE}/test_${TASK_NAME}"

complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name="test_${TASK_NAME}" \
    ++generation.task_name="$TASK_NAME" \
    ++generation.search.best_of_n.replicas=2 \
    ++ckpt_path="${PROTEINA_DIR}/ckpts" \
    ++ckpt_name=complexa_ligand.ckpt \
    ++autoencoder_ckpt_path="${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt" \
    ++hydra.run.dir="${OUTDIR}"

echo "=== Test complete ==="
echo "Output at: $OUTDIR"
find "$OUTDIR" -type f | head -20
