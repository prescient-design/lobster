# Running LBSTER Jobs with SLURM

This guide explains how to run a `lobster` training job using SLURM on a GPU-enabled system. It also describes which environment variables need to be exported for the job to run properly.

# SLURM Job Script
The provided example job script `scripts/train_ume.sh` is configured up for training the `UME` model on a GPU-enabled SLURM cluster. 


## Job Submission

### Submit the training job
```bash
sbatch slurm/scripts/train_ume.sh
```

## Environment Variables

The script sets up several  environment variables:
- `LOBSTER_RUNS_DIR`: S3 bucket for storing runs
- `LOBSTER_DATA_DIR`: Local data cache directory
- `NCCL_*`: Network communication settings for multi-GPU training
- `WANDB_*`: Weights & Biases logging configuration


### View job logs
Logs are written to `slurm/logs/train/` with format `{JOB_ID}_{JOB_NAME}.out`

## Running Evaluation Jobs

### Submit an evaluation job
```bash
sbatch slurm/scripts/evaluate_ume.sh 
```

### Evaluation Logs
Evaluation logs are written to `slurm/logs/eval/` with format `{JOB_ID}_{JOB_NAME}.out`
