#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem=256G
#SBATCH -o slurm/logs/%J.out
#SBATCH -t 00-24:00:00

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

# Define arrays for models and modalities
models=("ume-mini-base-12M" "ume-small-base-90M" "ume-medium-base-480M" "ume-large-base-740M")
modalities=("protein" "dna")

# Loop over all models and modalities
for model in "${models[@]}"; do
    for modality in "${modalities[@]}"; do
        echo "=========================================="
        echo "Evaluating model: $model"
        echo "Modality: $modality"
        echo "=========================================="
        
        # Set output directory based on model and modality
        output_dir="dgeb_results/${model}_${modality}"
        
        # Set max sequence length based on modality
        # DNA sequences can be longer, so use 8192 for DNA tasks
        if [ "$modality" = "dna" ]; then
            max_seq_length=8192
        else
            max_seq_length=1024
        fi
        
        ### DGEB EVAL
        srun -u --cpus-per-task 8 --cpu-bind=cores,verbose \
            uv run lobster_dgeb_eval \
            "$model" \
            --modality "$modality" \
            --output-dir "$output_dir" \
            --batch-size 32 \
            --max-seq-length "$max_seq_length" \
            --use-flash-attn \
            --l2-norm \
            --pool-type mean \
            --devices 0 \
            --seed 42
            
        echo "Completed evaluation for $model on $modality modality"
        echo "Results saved to: $output_dir"
        echo ""
    done
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
    


