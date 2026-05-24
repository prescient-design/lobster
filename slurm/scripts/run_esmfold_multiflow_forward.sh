#!/usr/bin/env bash
#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --qos llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task 16
#SBATCH --mem=128G
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_esmfold_mf.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval/%J_esmfold_mf.err
#SBATCH --job-name=esmfold_mf_fwd
#SBATCH -t 4:00:00

set -eo pipefail
mkdir -p /cv/scratch/u/lisanzas/slurm_logs/dplm2_eval

cd /cv/home/lisanzas/lobster

uv run python -m lobster.cmdline.esmfold_baseline \
    --config-path "../hydra_config/experiment" \
    --config-name esmfold_baseline \
    output_dir="./examples/esmfold_baseline_multiflow" \
    generation.input_structures="/cv/data/ai4dd/data2/lisanzas/multi_flow_data/test_set_filtered_pt/*.pt" \
    generation.batch_size=5 \
    generation.max_length=512
