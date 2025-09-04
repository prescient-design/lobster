#!/usr/bin/env bash

#SBATCH --job-name token_loss
#SBATCH --nodes 1 
#SBATCH --gpus-per-node 1 
#SBATCH --partition gpu2
#SBATCH --cpus-per-gpu 4 
#SBATCH --mem 150G
#SBATCH --time=1-00:00:00

source !/.bashrc
eval "$(mamba shell hook --shell bash)"

echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURMD_NODENAME = ${SLURMD_NODENAME}"
echo "SLURM_JOB_NUM_NODES = ${SLURM_JOB_NUM_NODES}"

# make sure that this is already set!
cd $LOBSTER_PROJECT_DIR

# use uv, which should already be set up
source .venv/bin/activate 

echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURMD_NODENAME = ${SLURMD_NODENAME}"
echo "SLURM_JOB_NUM_NODES = ${SLURM_JOB_NUM_NODES}"

nvidia-smi
mamba activate plaid 
mamba env list
echo $CONDA_PREFIX
which python

# see save_token_losses.py for the default parser arguments
srun torchrun token_selection/scripts/save_token_losses.py \
    --fasta_file  /data/bucket/freyn6/data/uniref50.fasta \
    --output_dir /data2/lux70/data/uniref50/per_token_losses \
    --max_num_per_shard 10000