#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=256G
#SBATCH -o slurm/logs/%J.out
#SBATCH -t 00-06:00:00

# srun hostname

nvidia-smi

source .venv/bin/activate
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

export WANDB_INSECURE_DISABLE_SSL=true
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# Tokenizer calls prior in UME callbacks prior to training
# cause issues. Disable if using callbacks, enable if not
export TOKENIZERS_PARALLELISM=true

# Sets default permissions to allow group write 
# access for newly created files. Remove if not needed
umask g+w 

### DGEB EVAL
srun -u --cpus-per-task 8 --cpu-bind=cores,verbose \
    uv run lobster_dgeb_eval \
    ume-mini-base-12M \
    --modality dna \
    --output-dir dgeb_results \
    --batch-size 32 \
    --max-seq-length 1024 \
    --use-flash-attn \
    --l2-norm \
    --pool-type mean \
    --devices 0 \
    --seed 42
    


