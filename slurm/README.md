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

### Sweeps
To run a sweep over hyperparameters, use hydra-submitit plugin in the following way:

```bash
lobster_train --multirun experiment=ume-2/small_molecule_slurm \
    model.lr=5e-6,1e-5,2e-5 \
    trainer.accumulate_grad_batches=1,2,4,8 \
    model.auxiliary_tasks.0.loss_weight=0.05,0.1,0.2

[2025-09-09 13:52:50,836][HYDRA] Submitit 'slurm' sweep output dir : multirun/2025-09-09/13-52-44
[2025-09-09 13:52:50,838][HYDRA]        #0 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=1 model.auxiliary_tasks.0.loss_weight=0.05
[2025-09-09 13:52:50,841][HYDRA]        #1 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=1 model.auxiliary_tasks.0.loss_weight=0.1
[2025-09-09 13:52:50,844][HYDRA]        #2 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=1 model.auxiliary_tasks.0.loss_weight=0.2
[2025-09-09 13:52:50,848][HYDRA]        #3 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=2 model.auxiliary_tasks.0.loss_weight=0.05
[2025-09-09 13:52:50,851][HYDRA]        #4 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=2 model.auxiliary_tasks.0.loss_weight=0.1
[2025-09-09 13:52:50,854][HYDRA]        #5 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=2 model.auxiliary_tasks.0.loss_weight=0.2
[2025-09-09 13:52:50,857][HYDRA]        #6 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=4 model.auxiliary_tasks.0.loss_weight=0.05
[2025-09-09 13:52:50,860][HYDRA]        #7 : experiment=ume-2/small_molecule_slurm model.lr=5e-06 trainer.accumulate_grad_batches=4 model.auxiliary_tasks.0.loss_weight=0.1
...
```


```
lobster_train --multirun experiment=ume-2/amino_acid \
    model.lr=1e-4,2e-4,5e-4 \
    trainer.accumulate_grad_batches=1,2
```

Each run is a slurm job with number of nodes and GPUs specified in the `small_molecule_slurm.yaml` file.
