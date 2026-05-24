# SLURM Scripts for ESM Atlas Structure Processing

This directory contains SLURM batch scripts for processing the ESM Atlas SimpleFold dataset (~37M structures) on a cluster.

## Cluster Constraints

**Available Partitions** (shared across all teams):
- **cpu**: 30 nodes with 128 vCPU / 256GB mem each
- **himem**: 3 nodes with 128 vCPU / 1TB mem each

## Available Scripts

### 1. Array Job - CPU Partition (⭐ **RECOMMENDED**)
**File**: `process_structures_esm_atlas_array.sh`

- **Partition**: cpu (30 nodes available)
- **Resources**: 128 CPUs, 240GB RAM per job
- **Time**: 24 hours
- **Array**: 10 parallel jobs
- **Best for**: Maximum speed with efficient resource utilization

```bash
sbatch slurm/scripts/process_structures_esm_atlas_array.sh
```

**Estimated time**: ~4-6 hours total (10 jobs × 128 CPUs = 1,280 CPUs)

**How it works**:
- ESM Atlas has directories `000` through `999`
- Each array job processes 100 directories (e.g., job 0 processes 000-099, job 1 processes 100-199, etc.)
- All 10 jobs run in parallel with maximum parallelization (128 workers each)
- Uses cpu partition which has 30 nodes available (plenty of capacity)

### 2. Array Job - HIMEM Partition (If CPU is busy)
**File**: `process_structures_esm_atlas_array_himem.sh`

- **Partition**: himem (**only 3 nodes** available)
- **Resources**: 128 CPUs, 1000GB RAM per job
- **Time**: 24 hours
- **Array**: 3 parallel jobs (limited by himem availability)
- **Best for**: When cpu partition is fully utilized

```bash
sbatch slurm/scripts/process_structures_esm_atlas_array_himem.sh
```

**Estimated time**: ~12-15 hours total (3 jobs × 128 CPUs = 384 CPUs)

**Note**: Each job processes ~333 directories to maximize the limited himem nodes.

### 3. Single-Node HIMEM
**File**: `process_structures_esm_atlas.sh`

- **Partition**: himem
- **Resources**: 128 CPUs, 1000GB RAM
- **Time**: 48 hours
- **Best for**: Single-job processing

```bash
sbatch slurm/scripts/process_structures_esm_atlas.sh
```

**Estimated time**: ~40-50 hours for all 37M files

### 4. Single-Node CPU
**File**: `process_structures_esm_atlas_cpu.sh`

- **Partition**: cpu
- **Resources**: 128 CPUs, 240GB RAM
- **Time**: 48 hours
- **Best for**: Simple single-node processing

```bash
sbatch slurm/scripts/process_structures_esm_atlas_cpu.sh
```

**Estimated time**: ~40-50 hours for all 37M files

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### Check array job status
```bash
squeue -u $USER -j <JOB_ID>
```

### View logs
```bash
# For single jobs
tail -f slurm/logs/process_esm_atlas_<JOB_ID>.out

# For array jobs
tail -f slurm/logs/process_esm_atlas_array_<JOB_ID>_<TASK_ID>.out
```

### Cancel jobs
```bash
# Cancel single job
scancel <JOB_ID>

# Cancel all array tasks
scancel <JOB_ID>

# Cancel specific array task
scancel <JOB_ID>_<TASK_ID>
```

## Output

All scripts save processed files to:
```
/data2/ume/simplefold_dataset/train_processed/
```

Failed files are logged to:
```
/data2/ume/simplefold_dataset/train_processed/failed_files.txt
```

## Resource Recommendations

### For ~37M files:

| Method | Partition | Nodes | Total CPUs | Est. Time | Best For |
|--------|-----------|-------|------------|-----------|----------|
| **Array CPU** ⭐ | cpu | 10 | 1,280 | **~4-6 hrs** | **Most efficient** |
| Array HIMEM | himem | 3 | 384 | ~12-15 hrs | When cpu busy |
| Single HIMEM | himem | 1 | 128 | ~40-50 hrs | Simple job |
| Single CPU | cpu | 1 | 128 | ~40-50 hrs | Simple job |

**Recommendation**: Use `process_structures_esm_atlas_array.sh` (CPU partition, 10 jobs) for fastest processing.

**Why CPU over HIMEM for array jobs?**
- cpu partition has 30 nodes available (can run 10 jobs easily)
- himem only has 3 nodes total (limits to 3 parallel jobs max)
- 256GB RAM per node is sufficient for structure processing
- 10 parallel jobs is 3.3× faster than 3 parallel jobs

## Customization

### Adjust array job count

**For CPU partition** (`process_structures_esm_atlas_array.sh`):
```bash
#SBATCH --array=0-9   # 10 jobs (100 dirs each) - OPTIMAL for 30-node partition
#SBATCH --array=0-19  # 20 jobs (50 dirs each) - more parallelism if nodes available
#SBATCH --array=0-4   # 5 jobs (200 dirs each) - fewer resources needed
```

**For HIMEM partition** (`process_structures_esm_atlas_array_himem.sh`):
```bash
#SBATCH --array=0-2   # 3 jobs MAXIMUM (himem only has 3 nodes!)
```

Don't forget to update `NUM_JOBS` variable in the script to match!

**Current optimal configuration**: 
- **CPU**: 10 jobs × 128 CPUs = 1,280 total CPUs
- **HIMEM**: 3 jobs × 128 CPUs = 384 total CPUs (limited by node availability)

### Adjust resources
Edit the SBATCH parameters:

```bash
#SBATCH --cpus-per-task=128  # Number of parallel workers
#SBATCH --mem=500G           # Memory allocation
#SBATCH --time=48:00:00      # Max runtime (HH:MM:SS)
```

### Adjust worker count
The `--num-workers` parameter in the script should generally match `--cpus-per-task`.

## Troubleshooting

### Job runs out of time
- Increase `--time` parameter
- Or use array jobs to split the work

### Out of memory errors
- Increase `--mem` parameter
- Or reduce `--num-workers`

### Job not starting
- Check queue: `squeue -u $USER`
- Check partition availability: `sinfo -p himem`
- Try a different partition (cpu instead of himem)

### Processing too slow
- Use array jobs for parallelization
- Increase `--cpus-per-task` and `--num-workers`

## Example Workflow

1. **Start with a test run** (process just a few files):
```bash
srun --nodes 1 --cpus-per-task 4 --mem=20G --time=1:00:00 --pty bash
uv run python scripts/process_structures.py \
    --input-dir /data2/ume/simplefold_dataset/esm_atlas/000/ \
    --output-dir /data2/ume/simplefold_dataset/train_processed/ \
    --num-workers 4 \
    --max-files 100 \
    --test
```

2. **Launch the full array job**:
```bash
sbatch slurm/scripts/process_structures_esm_atlas_array.sh
```

3. **Monitor progress**:
```bash
watch -n 30 'ls /data2/ume/simplefold_dataset/train_processed/*.pt | wc -l'
```

4. **Check for failures**:
```bash
cat /data2/ume/simplefold_dataset/train_processed/failed_files.txt
```

## Notes

- The script automatically skips already-processed files (unless `--overwrite` is used)
- Failed files are logged but don't stop processing
- Each `.pdb` file becomes one `.pt` file with the same base name
- Processed files are typically 10-50KB each (much smaller than source PDBs)

