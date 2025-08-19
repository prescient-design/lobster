#!/usr/bin/env bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH -p gpu2
#SBATCH --gpus 1
#SBATCH --time=48:00:00
#SBATCH -q preempt


#eval "$(conda shell.bash hook)"
#conda activate gsbm

source /homefs/home/lawrenh6/envlob/.venv/bin/activate

export OMP_NUM_THREADS=1;

export RUN_NAME=serious_chembl_large2; 

echo "Running on ${SLURM_JOB_NUM_NODES} nodes"
echo "Using ${SLURM_JOB_GPUS} GPUs per node"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURMD_NODENAME = ${SLURMD_NODENAME}"
echo "SLURM_JOB_NUM_NODES = ${SLURM_JOB_NUM_NODES}:29500"

# Check GPU availability
echo "=== GPU Availability ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Check PyTorch can see the GPUs

   python /homefs/home/lawrenh6/lobster/qm9_pair_gen/generate_pairs.py 
