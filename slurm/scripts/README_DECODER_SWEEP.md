# Decoder Dimension Sweep for Latent Generator (Protein-Ligand)

This sweep tests different decoder hidden dimensions combined with learning rate optimization.

## Overview

The sweep tests **4 decoder dimensions** × **2 learning rates** = **8 total training runs**

### Parameters Tested

**Decoder Dimensions:**
- 512
- 768
- 960
- 1024

**Learning Rates:**
- 1e-4
- 5e-4

**Important:** `struc_token_dim` and `ligand_struc_token_dim` are always kept equal for each run.

### Job Matrix

| Job # | Decoder Dim | Learning Rate | Resources |
|-------|-------------|---------------|-----------|
| 0     | 512         | 1e-4         | 8 GPUs, 1 node |
| 1     | 512         | 5e-4         | 8 GPUs, 1 node |
| 2     | 768         | 1e-4         | 8 GPUs, 1 node |
| 3     | 768         | 5e-4         | 8 GPUs, 1 node |
| 4     | 960         | 1e-4         | 8 GPUs, 1 node |
| 5     | 960         | 5e-4         | 8 GPUs, 1 node |
| 6     | 1024        | 1e-4         | 8 GPUs, 1 node |
| 7     | 1024        | 5e-4         | 8 GPUs, 1 node |

## Quick Start

### Run the Sweep

```bash
cd /homefs/home/lisanzas/scratch/Develop/lobster
bash slurm/scripts/train_latent_generator_protein_ligand_decoder_sweep.sh
```

**Important:** Do NOT use `sbatch`! Run directly with `bash`. Hydra handles SLURM submission.

## Configuration Details

### Training Parameters
- **Training steps**: 100,000
- **Warmup steps**: 10,000
- **Scheduler**: Cosine with warmup
- **Precision**: bf16-mixed
- **Gradient accumulation**: 16 batches
- **Strategy**: DDP with find_unused_parameters_true

### Model Architecture
- **Encoder dimension**: 256 (fixed)
- **Decoder dimensions**: 512, 768, 960, 1024 (swept)
- **Quantizer tokens**: 256 (fixed)
- **Both struc_token_dim and ligand_struc_token_dim**: Matched to decoder dimension

### Resource Allocation
- **Partition**: b200
- **QoS**: premium (high priority, no preemption)
- **GPUs per job**: 8
- **CPUs per task**: 16
- **Memory**: 256GB
- **Max runtime**: 7 days

## Monitoring

### Check SLURM Jobs

```bash
# View all your jobs
squeue -u $USER

# Count running jobs
squeue -u $USER | grep -c "R"

# View detailed info for a specific job
scontrol show job <JOB_ID>
```

### View Logs

Logs are stored in:
```
/data2/ume/latent_generator_/multirun/<date>/<time>/
├── 0/    # dim=512, lr=1e-4
├── 1/    # dim=512, lr=5e-4
├── 2/    # dim=768, lr=1e-4
├── 3/    # dim=768, lr=5e-4
├── 4/    # dim=960, lr=1e-4
├── 5/    # dim=960, lr=5e-4
├── 6/    # dim=1024, lr=1e-4
└── 7/    # dim=1024, lr=5e-4
```

Check training logs:
```bash
# View submitit stdout for job 0
tail -f /data2/ume/latent_generator_/multirun/<date>/<time>/0/.submitit/*/*/stdout

# Or navigate to job directory
cd /data2/ume/latent_generator_/multirun/<date>/<time>/0/
ls -la
```

### WandB Dashboard

All runs are grouped in WandB:
- **Project**: `lobster_latent_generator`
- **Group**: `latent_gen_decoder_sweep_<timestamp>`
- **Tags**: `decoder_sweep`, `latent_generator`, `protein_ligand`
- **Run names**: Include decoder dimension and learning rate (e.g., `lg_dim512_lr0.0001`)

Go to WandB and filter by the group to compare all 8 runs side-by-side.

## Output Structure

```
/data2/ume/latent_generator_/
├── multirun/
│   └── <date>/
│       └── <time>/
│           ├── 0/
│           │   ├── .submitit/           # SLURM job files
│           │   ├── .hydra/              # Hydra config
│           │   └── <timestamp>/         # Model checkpoints
│           ├── 1/
│           ├── ...
│           └── 7/
└── runs/
    └── <timestamp>/                     # Model checkpoints (from LOBSTER_RUNS_DIR)
```

## Analyzing Results

### Compare Model Sizes

After all jobs complete, compare in WandB:

1. **Training dynamics:**
   - Loss convergence speed
   - Training stability
   - Validation metrics

2. **Model capacity:**
   - How does decoder size affect final performance?
   - Diminishing returns at larger sizes?

3. **Learning rate interaction:**
   - Does optimal LR change with model size?
   - Larger models benefit from lower/higher LR?

4. **Resource efficiency:**
   - Training speed (steps/sec) vs model size
   - Memory usage vs model size

### Expected Insights

- **512**: Baseline, fastest training, smallest memory footprint
- **768**: Sweet spot between capacity and efficiency?
- **960**: Larger capacity, may show improvements on complex structures
- **1024**: Maximum capacity tested, check for overfitting

### Select Best Configuration

Consider:
- **Validation performance** (primary metric)
- **Training efficiency** (time to convergence)
- **Resource constraints** (memory, compute)
- **Inference speed** (for deployment)

## Troubleshooting

### Jobs Not Starting

Check resource availability:
```bash
sinfo -p b200
```

Check job status:
```bash
squeue -u $USER -t PENDING -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

### Out of Memory

If larger models (960, 1024) run out of memory:

1. Increase memory allocation (edit config):
```yaml
hydra:
  launcher:
    mem_gb: 384  # or 512
```

2. Reduce batch size (may need to adjust gradient accumulation):
```bash
# Add to multirun command:
trainer.accumulate_grad_batches=32
```

3. Reduce sequence length:
```bash
# Add to multirun command:
model.decoder_factory.decoder_mapping.vit_decoder.data_fixed_size=384
```

### Training Instability

If training shows instability (especially with lr=5e-4 on larger models):

- Check gradient norms in WandB
- May need gradient clipping adjustment
- Consider lower learning rate for largest models

### Cancel Jobs

Cancel specific sweep:
```bash
# Cancel all jobs in this sweep
scancel --name=train_latent_generator_protein_ligand_decoder_sweep

# Or cancel all your jobs
scancel -u $USER
```

## Customization

### Add More Dimensions

Edit the launch script to test additional sizes:

```bash
'model.decoder_factory.decoder_mapping.vit_decoder.struc_token_dim,model.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=256,512,768,1024,1280' \
model.optim.lr=1e-4,5e-4
```

This would create 5 × 2 = 10 jobs.

### Test More Learning Rates

```bash
'model.decoder_factory.decoder_mapping.vit_decoder.struc_token_dim,model.decoder_factory.decoder_mapping.vit_decoder.ligand_struc_token_dim=512,768,960,1024' \
model.optim.lr=5e-5,1e-4,2e-4,5e-4
```

This would create 4 × 4 = 16 jobs.

### Add Encoder Dimension Sweep

To also sweep encoder dimensions, add:

```bash
model.structure_encoder.embed_dim_hidden=128,256,512
```

This creates additional combinations.

## Next Steps

1. **Launch sweep**: `bash slurm/scripts/train_latent_generator_protein_ligand_decoder_sweep.sh`
2. **Monitor progress**: Check WandB dashboard regularly
3. **Analyze results**: Once complete, compare metrics in WandB
4. **Select best config**: Based on validation performance and efficiency
5. **Update production config**: Use winning combination for future training

## Notes

- All runs train for exactly 100,000 steps for fair comparison
- Using `premium` QoS ensures no preemption
- Paired decoder dimensions (protein and ligand) ensure architectural consistency
- WandB grouping makes comparison straightforward







