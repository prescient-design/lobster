#!/usr/bin/env bash

#SBATCH --partition llm-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_leflur_distogram_pairbias/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_leflur_distogram_pairbias/%J_%x.err
#SBATCH --mem=384G
#SBATCH --job-name=leflur_pairbias
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# PAIR-BIAS-ATTENTION finetune: distogram recipe + AF3/Proteina geometry-grounded pair-bias attention
# (decode current structure tokens -> Cb distance bins + relpos(±64) + chain-diff + hotspot -> per-layer
# per-head additive attention bias). Warm-starts from distogram step7680 (new to_bias zero-init ->
# step 0 == step7680). Self-conditioning stubbed off. Runs from the complex_infra WORKTREE.
#
# Usage:  sbatch slurm/scripts/train_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram_pairbias.sh
# Optional full-resume: RESUME_CKPT=<last.ckpt> sbatch ...

nvidia-smi
cd /cv/home/lisanzas/lobster/.claude/worktrees/complex_infra
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export WANDB_INIT_TIMEOUT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_NET_PLUGIN=""
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/leflur_distogram_pairbias/runs/"
export LOBSTER_DATA_DIR="/cv/scratch/u/lisanzas/.cache/"
export LOBSTER_USER=$(whoami)
export WANDB_BASE_URL=https://genentech.wandb.io

export HF_HOME="/cv/scratch/u/lisanzas/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/cv/scratch/u/lisanzas/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="/cv/scratch/u/lisanzas/.cache/huggingface"
export TORCH_HOME="/cv/scratch/u/lisanzas/.cache/torch"
export TORCH_NCCL_TRACE_BUFFER_SIZE=0
mkdir -p "$HF_HOME" "$TORCH_HOME"
export TOKENIZERS_PARALLELISM=true
umask g+w

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_leflur_distogram_pairbias
mkdir -p /cv/scratch/u/lisanzas/leflur_distogram_pairbias/runs

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

export HYDRA_OUT="/cv/scratch/u/lisanzas/hydra_outputs/distogram_pairbias"
mkdir -p "$HYDRA_OUT"

RESUME_ARGS=""
[ -n "${RESUME_CKPT:-}" ] && RESUME_ARGS="model.pretrained_ckpt=null model.ckpt_path=${RESUME_CKPT}"
uv run lobster_train \
    experiment=train_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram_pairbias \
    '+trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy' \
    '+trainer.strategy.find_unused_parameters=true' \
    '+trainer.strategy.timeout._target_=datetime.timedelta' \
    '+trainer.strategy.timeout.minutes=90' \
    ${RESUME_ARGS} \
    "hydra.run.dir=${HYDRA_OUT}/\${now:%Y-%m-%d}_\${now:%H-%M-%S}" \
    logger.name=leflur_distogram_pairbias-medium
