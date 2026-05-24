# Understanding `total_samples` vs `num_samples`

## TL;DR

`num_samples` in the config is **PER LENGTH**, not total structures!

## Example

### Your Current Config (`generate_unconditional.yaml`)
```yaml
generation:
  length: [100, 200, 300, 400, 500]  # 5 different lengths
  num_samples: 10                     # 10 samples PER LENGTH
```

**Normal execution**: Generates 10 × 5 = **50 structures**

### Distributed Generation

```bash
create_job_config.py --total_samples 1000 --samples_per_job 50
```

**What this means**:
- `total_samples = 1000` → 1000 samples **per length**
- `samples_per_job = 50` → each job generates 50 samples **per length**
- Number of jobs = 1000 / 50 = **20 jobs**

**Actual output per job**:
- Job 0: 50 samples × 5 lengths = **250 structures**
- Job 1: 50 samples × 5 lengths = **250 structures**
- ...
- Job 19: 50 samples × 5 lengths = **250 structures**

**Total across all jobs**: 1000 × 5 = **5,000 structures**

## The Math

```
actual_structures_per_job = samples_per_job × num_lengths
total_structures = total_samples × num_lengths
```

### With Your Current Config (5 lengths)
| total_samples | samples_per_job | jobs | structures/job | TOTAL STRUCTURES |
|---------------|-----------------|------|----------------|------------------|
| 100           | 10              | 10   | 50             | 500              |
| 500           | 25              | 20   | 125            | 2,500            |
| 1000          | 50              | 20   | 250            | 5,000            |
| 2000          | 100             | 20   | 500            | 10,000           |

### With Single Length Config
| total_samples | samples_per_job | jobs | structures/job | TOTAL STRUCTURES |
|---------------|-----------------|------|----------------|------------------|
| 100           | 10              | 10   | 10             | 100              |
| 500           | 25              | 20   | 25             | 500              |
| 1000          | 50              | 20   | 50             | 1,000            |
| 2000          | 100             | 20   | 100            | 2,000            |

## Recommendation

### For Simple 1:1 Mapping

Create a config variant with a **single length**:

```yaml
# generate_unconditional_single_length.yaml
generation:
  length: [500]  # SINGLE length only
  num_samples: 10  # Will be overridden
```

Then:
```bash
create_job_config.py \
    --total_samples 1000 \
    --samples_per_job 50 \
    --base_config src/lobster/hydra_config/experiment/generate_unconditional_single_length.yaml
```

Now `--total_samples 1000` = exactly **1000 structures** at length 500

### For Multi-Length Generation

If you want to keep multiple lengths, be aware of the multiplier:

**Goal: 1000 total structures across 5 lengths (200 per length)**
```bash
create_job_config.py --total_samples 200 --samples_per_job 10
# 20 jobs × 10 samples × 5 lengths = 1000 structures
```

**Goal: 1000 structures per length (5000 total)**
```bash
create_job_config.py --total_samples 1000 --samples_per_job 50
# 20 jobs × 50 samples × 5 lengths = 5000 structures
```

## Logging

The distributed generation script now logs this clearly:

```
Configuration for this job:
  Output: ./examples/generated_unconditional/job_0
  Samples per length: 50 (indices 0-50)
  Lengths: [100, 200, 300, 400, 500] (5 lengths)
  Total structures this job: 250
  Seed: 12345
  Steps: 1000
```

Pay attention to:
- **Samples per length**: The `num_samples` value (per length)
- **Lengths**: How many lengths
- **Total structures this job**: Actual structure count = samples_per_length × num_lengths

## Quick Reference

### Terminology
- **`total_samples`**: Total samples you want **per length** across all jobs (distributed generation concept)
- **`samples_per_job`**: Samples each job generates **per length** (distributed generation concept)
- **`num_samples`**: Samples to generate **per length** (hydra config parameter, gets overridden)
- **`length`**: List of lengths to generate at (hydra config parameter)

### Formula
```python
# Per job
samples_per_job = end_sample - start_sample
structures_per_job = samples_per_job × len(length)

# Total
total_jobs = total_samples / samples_per_job
total_structures = total_samples × len(length)
```

