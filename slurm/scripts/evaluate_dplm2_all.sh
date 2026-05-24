#!/usr/bin/env bash
# Evaluate DPLM-2 forward folding (CAMEO + MultiFlow) and inverse folding (MultiFlow)
# Submit AFTER dplm2_cameo_fwd and dplm2_mf_all jobs complete:
#   sbatch --dependency=afterok:<cameo_job>:<mf_job> slurm/scripts/evaluate_dplm2_all.sh

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_eval_all.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_eval_all.err
#SBATCH --job-name=eval_dplm2_all
#SBATCH -t 6:00:00

set -eo pipefail
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval

cd /cv/home/lisanzas/lobster

echo "=== 1/3: DPLM-2 Forward Folding on CAMEO ==="
uv run python scripts/evaluate_dplm2_forward_folding.py \
    --pred_pdb_dir /cv/home/lisanzas/dplm/generation-results/dplm2_650m_cameo/folding/pdb \
    --gt_data_dir /cv/data/ai4dd/data2/lisanzas/AFDB/valid_cameo_processed/ \
    --output dplm2_cameo_forward_folding_results.csv \
    --dataset cameo

echo "=== 2/3: DPLM-2 Forward Folding on MultiFlow ==="
uv run python scripts/evaluate_dplm2_forward_folding.py \
    --pred_pdb_dir /cv/home/lisanzas/dplm/generation-results/dplm2_650m_multiflow/folding/pdb \
    --gt_data_dir /cv/data/ai4dd/data2/lisanzas/multi_flow_data/test_set_filtered_pt/ \
    --output dplm2_multiflow_forward_folding_results.csv \
    --dataset multiflow

echo "=== 3/3: DPLM-2 Inverse Folding on MultiFlow (via ESMFold) ==="
uv run python scripts/evaluate_dplm2_esmfold.py \
    --dplm2_fasta /cv/home/lisanzas/dplm/generation-results/dplm2_650m_multiflow/inverse_folding/aatype.fasta \
    --gt_data_dir /cv/data/ai4dd/data2/lisanzas/multi_flow_data/test_set_filtered_pt/ \
    --output dplm2_multiflow_esmfold_results.csv \
    --device cuda

echo "=== All evaluations complete ==="
