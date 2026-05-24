#!/bin/bash
# Submit full training runs at different batch sizes
# Jobs that OOM will fail, giving us the max batch size
# Successful ones continue training normally

set -euo pipefail

LOBSTER_DIR="/cv/home/lisanzas/lobster"
LOG_DIR="/cv/scratch/u/lisanzas/slurm_logs/train_gen_ume_all"
RUNS_DIR="/cv/scratch/u/lisanzas/gen_ume_all/runs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

for BS in 36 48 64 96 128; do
    sbatch --partition=ai4dd-b200 --account=llm --qos=llm \
        --nodes=1 --ntasks-per-node=1 \
        --gres=gpu:b200:8 --cpus-per-task=48 --mem=256G \
        -t 7-00:00:00 \
        --job-name="genume-all-bs${BS}" \
        -o "${LOG_DIR}/bs${BS}_%j.out" \
        -e "${LOG_DIR}/bs${BS}_%j.err" \
        --wrap="
cd ${LOBSTER_DIR}
nvidia-smi
export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_NET_PLUGIN=\"\"
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export LOBSTER_RUNS_DIR=${RUNS_DIR}
export LOBSTER_DATA_DIR=/cv/scratch/u/lisanzas/.cache/
export LOBSTER_USER=\$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io
export TOKENIZERS_PARALLELISM=true
unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

uv run lobster_train \
    experiment=train_gen_ume_protein_ligand \
    data=structure_ligand_all \
    model.encoder_kwargs.model_size=medium \
    model.lr=1e-3 \
    model.num_warmup_steps=2500 \
    model.num_training_steps=50000 \
    model.scheduler_kwargs.num_warmup_steps=2500 \
    model.scheduler_kwargs.num_training_steps=50000 \
    data.batch_size=${BS} \
    data.num_workers=8 \
    trainer.devices=8 \
    trainer.accumulate_grad_batches=20 \
    trainer.max_steps=50000 \
    trainer.val_check_interval=500 \
    +trainer.strategy=ddp_find_unused_parameters_true \
    trainer.num_sanity_val_steps=2 \
    trainer.precision=bf16-mixed \
    model.use_se3_augmentation=true \
    model.se3_translation_scale=1.0 \
    callbacks.protein_ligand_decode.minimize_ligand=true \
    callbacks.protein_ligand_inverse_folding.minimize_ligand=true \
    callbacks.protein_ligand_forward_folding.minimize_ligand=true \
    logger.name=gen_ume_all-medium_bs${BS}_lr1e-3

echo 'COMPLETED: batch_size=${BS}'
"

    echo "Submitted: batch_size=${BS} (effective: $((BS * 8 * 20)))"
done
