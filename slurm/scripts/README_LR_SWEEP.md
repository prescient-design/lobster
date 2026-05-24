# Learning Rate Sweep for Latent Generator (Protein-Ligand)

This directory contains scripts to run a hyperparameter sweep over learning rates for the latent generator model with protein-ligand training.

## Setup Overview

The sweep uses Hydra's submitit launcher to automatically submit separate SLURM jobs for each learning rate configuration.

### Files Created

1. **Experiment Config**: `src/lobster/hydra_config/experiment/train_latent_generator_protein_ligand_slurm.yaml`
   - Extends the base `train_latent_generator` config
   - Adds SLURM launcher configuration
   - Sets training steps to 100,000
   - Configures WandB grouping for sweep runs

2. **Launch Script**: `slurm/scripts/train_latent_generator_protein_ligand_lr_sweep.sh`
   - Launches the multirun sweep
   - Tests 4 learning rates: 5e-3, 1e-3, 5e-4, 1e-4

## Quick Start

### Run the Sweep

```bash
cd /homefs/home/lisanzas/scratch/Develop/lobster
bash slurm/scripts/train_latent_generator_protein_ligand_lr_sweep.sh
```

**Important**: Do NOT use `sbatch` on this script! Just run it directly with `bash`. Hydra will handle submitting the SLURM jobs.

### What Happens

1. Hydra creates 4 separate SLURM jobs (one per learning rate)
2. Each job gets:
   - **1 node**
   - **8 GPUs**
   - **16 CPUs per task**
   - **256GB memory**
   - **7 days max runtime**
   - **Partition**: b200
   - **QoS**: preempt

3. All jobs are grouped in WandB under the same sweep group for easy comparison

## Monitoring

### Check SLURM Jobs

```bash
# View all your jobs
squeue -u $USER

# View detailed job info
scontrol show job <JOB_ID>

# View job logs (Hydra creates these automatically)
ls -la multirun/<timestamp>/<job_num>/
```

### WandB Dashboard

All runs are logged to WandB:
- **Project**: `lobster_latent_generator`
- **Group**: `latent_gen_lr_sweep_<timestamp>`
- **Tags**: `lr_sweep`, `latent_generator`, `protein_ligand`

Go to your WandB dashboard and filter by the group to compare all 4 runs.

## Configuration Details

### Training Parameters

- **Training steps**: 100,000
- **Warmup steps**: 10,000
- **Learning rates tested**: 5e-3, 1e-3, 5e-4, 1e-4
- **Scheduler**: Cosine with warmup
- **Precision**: bf16-mixed
- **Gradient accumulation**: 16 batches
- **Strategy**: DDP with find_unused_parameters_true

### Data Configuration

- **Dataset**: structure_ligand_pdb
- **Model**: latent_generator_ligand
- **Workers**: 8 per GPU

### Resource Allocation

Each learning rate run gets a full 8-GPU node to ensure consistent training speed and fair comparison.

## Output Structure

```
multirun/
└── <timestamp>/
    ├── 0/                    # lr=5e-3
    │   ├── .submitit/
    │   ├── train.log
    │   └── <timestamp>/      # Model checkpoints
    ├── 1/                    # lr=1e-3
    ├── 2/                    # lr=5e-4
    └── 3/                    # lr=1e-4
```

## Customizing the Sweep

### Change Learning Rates

Edit the launch script and modify the `model.optim.lr` parameter:

```bash
lobster_train --multirun \
    experiment=train_latent_generator_protein_ligand_slurm \
    model.optim.lr=1e-4,5e-5,1e-5,5e-6  # Your custom values
```

### Add More Hyperparameters

You can sweep over multiple parameters simultaneously:

```bash
lobster_train --multirun \
    experiment=train_latent_generator_protein_ligand_slurm \
    model.optim.lr=1e-4,5e-5,1e-5 \
    trainer.accumulate_grad_batches=8,16,32 \
    model.num_warmup_steps=5000,10000,20000
```

This creates a Cartesian product: 3 × 3 × 3 = **27 total jobs**

### Adjust Resource Allocation

To use fewer GPUs per job (allowing more parallel jobs):

Edit `train_latent_generator_protein_ligand_slurm.yaml`:

```yaml
hydra:
  launcher:
    gpus_per_node: 4  # Instead of 8
    tasks_per_node: 4
    cpus_per_task: 16
```

## Troubleshooting

### Jobs Not Submitting

- Check you have permissions on the b200 partition
- Verify your SLURM allocation has available resources
- Check job queue: `squeue -u $USER`

### WandB Not Logging

- Verify `WANDB_BASE_URL` is set correctly
- Check WandB credentials: `wandb login`
- Look for WandB errors in job logs

### Out of Memory

If jobs fail with OOM errors:
- Reduce `trainer.accumulate_grad_batches` in the config
- Adjust data batch size if needed
- Request more memory: `hydra.launcher.mem_gb=512`

## Analyzing Results

Once all jobs complete, compare in WandB:

1. Go to your project: `lobster_latent_generator`
2. Filter by group: `latent_gen_lr_sweep_<timestamp>`
3. Compare metrics:
   - Training loss curves
   - Validation metrics
   - Convergence speed
4. Select the best learning rate based on final validation performance

## Cancel All Jobs

If you need to stop the sweep:

```bash
# Cancel all your jobs
scancel -u $USER

# Or cancel specific sweep jobs by name
scancel --name=train_latent_generator_protein_ligand_slurm
```

