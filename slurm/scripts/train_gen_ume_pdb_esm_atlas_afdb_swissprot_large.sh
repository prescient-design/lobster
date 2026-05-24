#!/usr/bin/env bash

#SBATCH --partition b200
#SBATCH --nodes 10
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --cpus-per-task 24
#SBATCH -o /data2/lisanzas/gen_ume/slurm/logs/train/%J_%x.out
#SBATCH -q preempt
#SBATCH --mem=1024G
#SBATCH --job-name=gen_ume_pdb_afdb
#SBATCH -t 7-00:00:00
#SBATCH -q premium

nvidia-smi

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/opt/amazon/ofi-nccl/lib64

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO

export LOBSTER_RUNS_DIR="/data2/lisanzas/gen_ume/runs/"
export LOBSTER_DATA_DIR="/data2/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write
# access for newly created files. Remove if not needed
umask g+w

srun -u --cpus-per-task $SLURM_CPUS_PER_TASK --cpu-bind=cores,verbose \
    lobster_train \
    experiment=train_gen_ume \
    data=structure_esm_atlas_afdb_swissprot \
    data.num_workers=8 \
    +data.stat_workers=16 \
    +model.latent_generator_model_name="LG full attention" \
    model.encoder_kwargs.model_size=large \
    callbacks=gen_ume_full_eval \
    trainer.num_sanity_val_steps=0 \
    trainer.val_check_interval=1000 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES \




