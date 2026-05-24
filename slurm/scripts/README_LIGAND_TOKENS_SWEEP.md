# Ligand Token Count Sweep for Latent Generator (Protein-Ligand)

This sweep tests different numbers of ligand tokens in the quantizer with fixed decoder dimension and learning rate.

## Overview

The sweep tests **5 ligand token counts** = **5 total training runs**

### Fixed Parameters

- **Decoder dimension**: 512 (both protein and ligand)
- **Learning rate**: 5e-4
- **Training steps**: 100,000
- **Warmup steps**: 10,000

### Parameters Tested

**Ligand Token Counts:**
- 256
- 512 (baseline)
- 1024
- 2048
- 4096

### Job Matrix

| Job # | Ligand Tokens | Decoder Dim | Learning Rate | Resources |
|-------|---------------|-------------|---------------|-----------|
| 0     | 256          | 512         | 5e-4         | 8 GPUs, 1 node |
| 1     | 512          | 512         | 5e-4         | 8 GPUs, 1 node |
| 2     | 1024         | 512         | 5e-4         | 8 GPUs, 1 node |
| 3     | 2048         | 512         | 5e-4         | 8 GPUs, 1 node |
| 4     | 4096         | 512         | 5e-4         | 8 GPUs, 1 node |

## Quick Start

### Run the Sweep

```bash
cd /homefs/home/lisanzas/scratch/Develop/lobster
bash slurm/scripts/train_latent_generator_protein_ligand_tokens_sweep.sh
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
- **Decoder dimensions**: 512 (fixed for both protein and ligand)
- **Protein quantizer tokens**: 256 (fixed)
- **Ligand quantizer tokens**: 256, 512, 1024, 2048, 4096 (swept)

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
├── 0/    # 256 tokens
├── 1/    # 512 tokens (baseline)
├── 2/    # 1024 tokens
├── 3/    # 2048 tokens
└── 4/    # 4096 tokens
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
- **Group**: `latent_gen_ligand_tokens_sweep_<timestamp>`
- **Tags**: `ligand_tokens_sweep`, `latent_generator`, `protein_ligand`
- **Run names**: Include token count (e.g., `lg_tokens512_dim512_lr5e-4`)

Go to WandB and filter by the group to compare all 5 runs side-by-side.

## Purpose & Analysis

### Research Questions

This sweep answers:

1. **Capacity vs Efficiency**: How does ligand representation capacity affect model performance?
2. **Optimal Tokenization**: What's the sweet spot for ligand token count?
3. **Diminishing Returns**: At what point does adding more tokens stop improving performance?
4. **Memory/Speed Trade-offs**: How do larger token counts affect training efficiency?

### Expected Insights

- **256 tokens**: Minimal representation, fast but potentially underfitting ligand diversity
- **512 tokens (baseline)**: Current default, establishes performance baseline
- **1024 tokens**: Increased capacity, may improve on complex ligand structures
- **2048 tokens**: High capacity, test if more expressiveness helps
- **4096 tokens**: Maximum capacity tested, check for overfitting or diminishing returns

### What to Look For

1. **Training Dynamics:**
   - Does loss converge faster/slower with different token counts?
   - Are there stability issues at extreme values?

2. **Final Performance:**
   - Validation metrics (reconstruction quality, etc.)
   - Does performance plateau at some token count?

3. **Efficiency:**
   - Training speed (steps/sec) vs token count
   - Memory usage vs token count
   - Is the performance gain worth the computational cost?

4. **Ligand Reconstruction Quality:**
   - Especially important for this sweep
   - Check ligand-specific loss terms
   - Visual inspection of reconstructed ligands

### Selecting Best Configuration

Consider:
- **Performance**: Primary metric (validation loss, reconstruction quality)
- **Efficiency**: Training speed and memory usage
- **Generalization**: Check for overfitting with larger token counts
- **Practical constraints**: Deployment memory/compute budgets

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

Larger token counts (2048, 4096) may require more memory:

1. Increase memory allocation:
```yaml
hydra:
  launcher:
    mem_gb: 384  # or 512
```

2. Reduce batch size or increase gradient accumulation:
```bash
# Add to config or command line
trainer.accumulate_grad_batches=32
```

3. Monitor memory usage in WandB for each run

### Training Instability

If larger token counts show instability:

- Check gradient norms
- May need gradient clipping adjustment
- Consider lower learning rate for extreme token counts

### Unexpectedly Slow Training

Large token counts increase computation:

- Monitor training speed (steps/sec) in logs
- Compare across different token counts
- May be expected - document the trade-off

### Cancel Jobs

Cancel specific sweep:
```bash
# Cancel all jobs in this sweep
scancel --name=train_latent_generator_protein_ligand_tokens_sweep

# Or cancel all your jobs
scancel -u $USER
```

## Customization

### Test Additional Token Counts

Edit the launch script:

```bash
model.quantizer.ligand_n_tokens=128,256,512,1024,2048,4096,8192
```

### Change Fixed Parameters

Edit the experiment config to test different combinations:

```yaml
model:
  optim:
    lr: 1e-4  # Different learning rate
  decoder_factory:
    decoder_mapping:
      vit_decoder:
        struc_token_dim: 768  # Different decoder size
        ligand_struc_token_dim: 768
```

### Also Sweep Protein Tokens

Add protein token sweep to the command:

```bash
lobster_train --multirun \
    experiment=train_latent_generator_protein_ligand_tokens_sweep \
    model.quantizer.ligand_n_tokens=256,512,1024,2048,4096 \
    model.quantizer.n_tokens=128,256,512
```

This creates 5 × 3 = 15 jobs.

## Expected Results

### Computational Cost Scaling

Token count primarily affects:
1. **Quantizer computation**: Linear scaling with token count
2. **Memory usage**: Increases with codebook size
3. **Training speed**: May decrease slightly with larger codebooks

### Performance Expectations

Typical behavior:
- **Underfitting zone** (< optimal): Performance improves rapidly with more tokens
- **Optimal zone**: Performance plateaus, best cost/benefit ratio
- **Diminishing returns** (> optimal): Minimal improvement, increased cost

### Validation Metrics

Key metrics to compare:
- Overall reconstruction loss
- Ligand-specific losses (`ligand_l2_loss`, `ligand_pairwise_l2_loss`)
- Convergence rate
- Final model quality

## Next Steps

1. **Launch sweep**: `bash slurm/scripts/train_latent_generator_protein_ligand_tokens_sweep.sh`
2. **Monitor progress**: Check WandB dashboard regularly
3. **Compare metrics**: Focus on ligand-specific reconstruction quality
4. **Analyze trade-offs**: Performance vs efficiency
5. **Select optimal count**: Based on validation performance and computational constraints
6. **Update configs**: Use winning token count for future training

## Notes

- All runs train for exactly 100,000 steps for fair comparison
- Using `premium` QoS ensures no preemption
- Fixed decoder dim and LR isolates the effect of token count
- WandB grouping makes comparison straightforward
- This sweep complements the decoder dimension sweep








