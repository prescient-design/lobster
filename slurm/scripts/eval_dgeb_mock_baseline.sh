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

# Define arrays for modalities
modalities=("protein" "dna")

# Loop over all modalities
for modality in "${modalities[@]}"; do
    echo "=========================================="
    echo "Running mock baseline evaluation"
    echo "Modality: $modality"
    echo "=========================================="
    
    # Set output directory based on modality
    output_dir="dgeb_mock_results/mock_baseline_${modality}"
    
    # Set max sequence length based on modality
    # DNA sequences can be longer, so use 8192 for DNA tasks
    if [ "$modality" = "dna" ]; then
        max_seq_length=8192
    else
        max_seq_length=1024
    fi
    
    ### DGEB MOCK EVAL
    srun -u --cpus-per-task 8 --cpu-bind=cores,verbose \
        uv run lobster_dgeb_mock_eval \
        --model-name "mock-ume-baseline" \
        --modality "$modality" \
        --output-dir "$output_dir" \
        --batch-size 32 \
        --max-seq-length "$max_seq_length" \
        --l2-norm \
        --pool-type mean \
        --seed 42 \
        --embed-dim 768 \
        --num-layers 12
        
    echo "Completed mock baseline evaluation for $modality modality"
    echo "Results saved to: $output_dir"
    echo ""
done

echo "=========================================="
echo "All mock baseline evaluations completed!"
echo "==========================================" 