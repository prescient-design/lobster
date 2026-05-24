#!/bin/bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 24
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/proteina_gen_logs/test_%J.out
#SBATCH -e /cv/scratch/u/lisanzas/proteina_gen_logs/test_%J.err
#SBATCH --job-name=proteina-test
#SBATCH -t 01:00:00

set -euo pipefail

PROTEINA_DIR="/cv/scratch/u/lisanzas/proteina-complexa"
cd "$PROTEINA_DIR"
source .venv/bin/activate
source env.sh

echo "=== Testing Proteina-Complexa ligand binder generation ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run a small test: 2 replicas of the default ligand target (7V11)
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name="test_ligand_gen" \
    ++generation.task_name="39_7V11_LIGAND" \
    ++generation.search.best_of_n.replicas=2 \
    ++ckpt_path="${PROTEINA_DIR}/ckpts" \
    ++ckpt_name=complexa_ligand.ckpt \
    ++autoencoder_ckpt_path="${PROTEINA_DIR}/ckpts/complexa_ligand_ae.ckpt"

echo "=== Test complete ==="
