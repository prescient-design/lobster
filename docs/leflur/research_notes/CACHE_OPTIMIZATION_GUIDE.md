# Cache Building Optimization Guide - For Network Filesystems

## Problem Analysis

Your output shows:
```
Stage 1: 38M paths in 805s (47k paths/s) ✓ Good
Stage 2: 1,068 files/s with 192 workers ✗ VERY SLOW (9+ hours estimated)
```

**Root cause**: Network filesystem (NFS/Lustre) saturated with random stat() calls.

---

## 🚀 RECOMMENDED SOLUTIONS (Fastest to Slowest)

### Option 1: Skip Stat Calls Entirely ⚡ **FASTEST** (Recommended)

**Use this if**: Files are known to exist and be valid.

```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    skip_stat=True,  # Skip all stat calls during cache build
)
```

**Performance**:
- Stage 1: ~13 minutes (path discovery)
- Stage 2: **Instant** (no stat calls)
- **Total: ~13 minutes** vs 9+ hours

**Trade-off**: Files validated on first access instead of during cache build.

---

### Option 2: Use System `find` Command ⚡⚡ **VERY FAST**

**Use this if**: You have access to Unix `find` command.

```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    use_find_command=True,  # Use system find (5-10x faster)
    skip_stat=True,  # Also skip stat for maximum speed
)
```

**Performance**:
- Stage 1: **2-5 minutes** (find is optimized for this)
- Stage 2: Instant (if skip_stat=True)
- **Total: ~2-5 minutes**

**Why faster?**: `find` is written in C and optimized for filesystem traversal.

---

### Option 3: Reduce Worker Count (Network FS)

**Use this if**: You must do stat calls on network filesystem.

```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    stat_workers=16,  # Reduce from 192 to avoid overwhelming NFS
)
```

**Performance**:
- Stage 1: ~13 minutes
- Stage 2: ~1-3 hours (much better than 9 hours)
- **Total: ~1.5-3.5 hours**

**Worker recommendations by filesystem**:
- **Local SSD**: 64-128 workers
- **Local HDD**: 32-64 workers
- **NFS/Network**: 8-32 workers
- **Slow network**: 4-16 workers

---

### Option 4: Combine Multiple Optimizations ⚡⚡⚡ **ULTIMATE**

```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    use_find_command=True,  # Fast discovery
    skip_stat=True,          # Skip validation
    cache_file="/fast/local/ssd/cache.parquet",  # Local SSD for cache
)
```

**Performance**: **~2-5 minutes total** 🔥

---

## 📊 Performance Comparison Table

| Method | Stage 1 | Stage 2 | Total | Risk |
|--------|---------|---------|-------|------|
| Current (192 workers) | 13 min | 9+ hours | **9+ hours** | Low |
| Reduced workers (16) | 13 min | 1-3 hours | **1.5-3.5 hours** | Low |
| Skip stat only | 13 min | instant | **13 min** | Medium* |
| Find + skip stat | 2-5 min | instant | **2-5 min** | Medium* |
| Local cache location | 13 min | 30-60 min | **45-75 min** | Low |

\* *Risk: Invalid files detected on first access instead of cache build*

---

## 🔧 Additional Optimizations

### 5. Cache on Local SSD

Store cache on fast local storage:

```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",  # Slow NFS
    cache_file="/tmp/structure_cache.parquet",  # Fast local
)
```

### 6. Use Lustre Optimizations (if available)

```bash
# Set stripe count before scanning
lfs setstripe -c 4 /data2/ume/simplefold_dataset/train_processed/.cache/

# Then run with reduced workers
```

```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    stat_workers=32,  # Lustre handles more workers better than NFS
)
```

### 7. Sample-Based Validation

If you want some validation but not full stat:

```python
# Modify the code to only stat a random sample (e.g., 1%)
# This validates data quality without full scan
```

### 8. Two-Phase Approach

Build quick cache now, enhance later:

```python
# Phase 1: Quick cache (minutes)
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    skip_stat=True,
)

# Use dataset immediately...

# Phase 2: Background validation (optional, run during training)
# Add stat data incrementally during idle time
```

---

## 🎯 RECOMMENDED WORKFLOW

### For Your 38M File Dataset on NFS:

```python
# STEP 1: Build cache with skip_stat (13 minutes)
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    skip_stat=True,  # Skip stat during build
    use_find_command=True,  # Use find if available (2-5 min)
)

# STEP 2: Start training immediately
# Files will be validated on access (negligible overhead)

# STEP 3 (Optional): Rebuild with stat during next downtime
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    rebuild_cache=True,
    stat_workers=16,  # Reduced for NFS
)
```

---

## ⚠️ Understanding the Trade-offs

### Why is stat() so slow on your filesystem?

1. **Network latency**: Each stat call = network round-trip
2. **Random I/O**: 38M random stat calls kill performance
3. **Filesystem metadata**: NFS not optimized for this workload
4. **Parallel saturation**: 192 workers overwhelm network

### Why skip_stat is safe:

1. **Files validated on access**: torch.load() will fail if file is bad
2. **DataLoader retries**: Most training loops handle occasional failures
3. **Files rarely corrupt**: On read-only datasets, files don't change
4. **Quick detection**: Bad files detected within first epoch

### When NOT to use skip_stat:

- Actively writing new files
- Unreliable storage (frequent corruption)
- Need exact file count before training
- Compliance/audit requirements

---

## 💡 Pro Tips

### 1. Check Your Filesystem Type

```bash
df -T /data2/ume/simplefold_dataset/train_processed
```

Different filesystems need different worker counts:
- **NFS**: 8-16 workers
- **Lustre**: 32-64 workers
- **Local**: 64-128 workers

### 2. Test Worker Count

```python
# Test different worker counts on a subset
for workers in [4, 8, 16, 32, 64]:
    start = time.time()
    dataset = StructureDataset(
        root="/data2/ume/simplefold_dataset/train_processed",
        stat_workers=workers,
        testing=True,  # Limit to 500 files
    )
    print(f"{workers} workers: {time.time() - start:.2f}s")
```

### 3. Monitor Network Usage

```bash
# While cache is building
watch -n 1 'nfsstat -c'
```

If you see high retransmits or timeouts, reduce workers.

### 4. Use Screen/Tmux

Cache building takes time, use a persistent session:

```bash
screen -S cache_build
python build_cache.py
# Ctrl-A, D to detach
```

---

## 📝 Example Scripts

### Quick Start (Recommended)

```python
#!/usr/bin/env python
"""Build cache quickly for 38M files."""

from lobster.datasets import StructureDataset

# Fastest method: Skip stat + find command
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    skip_stat=True,           # Skip validation (fast!)
    use_find_command=True,    # Use system find (faster!)
)

print(f"Cache built! Found {len(dataset)} structures")
```

### Conservative (With Validation)

```python
#!/usr/bin/env python
"""Build cache with validation for network filesystem."""

from lobster.datasets import StructureDataset

# Slower but validates all files
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    stat_workers=16,  # Reduced for NFS
)

print(f"Cache built with validation! Found {len(dataset)} structures")
```

### Development/Testing

```python
#!/usr/bin/env python
"""Test cache building performance."""

from lobster.datasets import StructureDataset
import time

# Test on small subset first
start = time.time()
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    testing=True,  # Limit to 500 files
    skip_stat=True,
)
elapsed = time.time() - start

# Extrapolate to full dataset
full_time_estimate = elapsed * (38_117_657 / 500) / 60
print(f"Test took {elapsed:.2f}s")
print(f"Full dataset estimated: {full_time_estimate:.1f} minutes")
```

---

## 🎓 Summary

**For your specific case (38M files on `/data2/` NFS):**

### ⭐ **BEST SOLUTION**:
```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    skip_stat=True,           # Skip stat calls
    use_find_command=True,    # Use system find
)
```
**Time: 2-5 minutes** (vs 9+ hours) ⚡⚡⚡

### Good Alternative:
```python
dataset = StructureDataset(
    root="/data2/ume/simplefold_dataset/train_processed",
    stat_workers=16,  # Reduce from 192
)
```
**Time: 1.5-3.5 hours** (vs 9+ hours) ⚡

Choose based on your needs for validation vs speed!












