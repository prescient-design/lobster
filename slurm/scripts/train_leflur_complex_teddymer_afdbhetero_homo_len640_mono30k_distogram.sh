#!/usr/bin/env bash

#SBATCH --partition ai4dd-b200
#SBATCH --account llm
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:b200:8
#SBATCH --cpus-per-task 48
#SBATCH -o /cv/scratch/u/lisanzas/slurm_logs/train_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram/%J_%x.out
#SBATCH -e /cv/scratch/u/lisanzas/slurm_logs/train_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram/%J_%x.err
#SBATCH --mem=384G
#SBATCH --job-name=leflur_homo_distogram
#SBATCH -t 7-00:00:00
#SBATCH -q llm

# DISTOGRAM-AUX finetune: identical recipe to the homo+mono30k run, plus an auxiliary AF3-style
# distogram head + loss (binned Cb-Cb distance map, full intra + inter chain, inter-chain
# up-weighted) to sharpen the docking representation. Warm-starts from the FROZEN base-homo
# last.ckpt (= the DockQ eval baseline), so it's a clean A/B on the distogram head.
#
# NOTE: runs from the complex_infra WORKTREE (the distogram code + config live there).
#
# Usage:
#   sbatch slurm/scripts/train_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram.sh

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

export LOBSTER_RUNS_DIR="/cv/scratch/u/lisanzas/leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram/runs/"
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

mkdir -p /cv/scratch/u/lisanzas/slurm_logs/train_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram
mkdir -p /cv/scratch/u/lisanzas/leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram/runs

unset SLURM_NTASKS
unset SLURM_JOB_NAME
unset SLURM_NTASKS_PER_NODE

export HYDRA_OUT="/cv/scratch/u/lisanzas/hydra_outputs/homo_mono30k_distogram"
mkdir -p "$HYDRA_OUT"

# Optional full-resume after a crash: pass RESUME_CKPT=<last.ckpt>. Unset => fresh launch.
RESUME_ARGS=""
[ -n "${RESUME_CKPT:-}" ] && RESUME_ARGS="model.pretrained_ckpt=null model.ckpt_path=${RESUME_CKPT}"
# Optional distogram loss-weight override (default set in the experiment config = 0.5).
WEIGHT_ARGS=""
[ -n "${DISTO_W:-}" ] && WEIGHT_ARGS="model.distogram_loss_weight=${DISTO_W}"
uv run lobster_train \
    experiment=train_leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram \
    '+trainer.strategy._target_=lightning.pytorch.strategies.DDPStrategy' \
    '+trainer.strategy.find_unused_parameters=true' \
    '+trainer.strategy.timeout._target_=datetime.timedelta' \
    '+trainer.strategy.timeout.minutes=90' \
    ${RESUME_ARGS} \
    ${WEIGHT_ARGS} \
    "hydra.run.dir=${HYDRA_OUT}/\${now:%Y-%m-%d}_\${now:%H-%M-%S}" \
    logger.name=leflur_complex_teddymer_afdbhetero_homo_len640_mono30k_distogram-medium
