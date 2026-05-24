#!/usr/bin/env bash

#SBATCH --partition ai4dd
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_pdb_only/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_pdb_only/%J_%x.err
#SBATCH --mem=256G
#SBATCH --job-name=gen_ume_pl_pdb
#SBATCH -t 3-00:00:00
#SBATCH -q llm

# Gen-UME Protein-Ligand Model - PDB Only Data (Sanity Check)
# Trains the protein-ligand model architecture but with protein-only data
# to verify training works without mixed batch complexity

nvidia-smi

cd /cv/home/lisanzas/lobster
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_protein_ligand_pdb_only/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true
umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_protein_ligand_pdb_only

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

# Use same experiment but override data to protein-only via command line
uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    'data.path_to_datasets=["/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/train.pt","/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/val.pt","/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/split_data/test.pt"]' \
    'data.dataset_types=["structure","structure","structure"]' \
    'data.cluster_file_list=["/cv/data/ai4dd/data2/lisanzas/latent_generator_files/pdb_data/pdb_seqid40_clusters.pt",null,null]' \
    'data.files_to_keep_list=[null,null,null]' \
    data.use_ligand_dataset=false \
    data.datasets=pdb_only \
    data.batch_size=16 \
    logger.name=gen_ume_protein_ligand_pdb_only \
    trainer.max_steps=50000 \
    trainer.devices=8 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2
