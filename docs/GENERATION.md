# Distributed Generation with Hydra Submitit

## Basic Usage

Run generation across multiple GPUs using Hydra's submitit launcher:

```bash
uv run python -m lobster.cmdline.generate \
  --multirun \
  experiment=generate_unconditional \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=b200 \
  hydra.launcher.qos=preempt \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.cpus_per_task=32 \
  hydra.launcher.mem_gb=128 \
  hydra.launcher.timeout_min=1440 \
  generation.num_samples=2 \
  generation.length=[100] \
  generation.nsteps=50 \
  seed=12345
```

## Wandb Integration

Track all runs from a multirun submission with a shared unique tag:

```bash
uv run python -m lobster.cmdline.generate \
  --multirun \
  experiment=generate_unconditional \
  hydra/launcher=submitit_slurm \
  hydra.launcher.partition=b200 \
  hydra.launcher.qos=preempt \
  hydra.launcher.gpus_per_node=1 \
  hydra.launcher.cpus_per_task=32 \
  hydra.launcher.mem_gb=128 \
  hydra.launcher.timeout_min=1440 \
  generation.num_samples=2 \
  generation.length=[100] \
  generation.nsteps=50 \
  seed=12345 \
  wandb.enabled=true \
  wandb.project=lobster-generation \
  wandb.entity=your-wandb-entity
```

All runs in the multirun will automatically share:
- **Group**: All jobs grouped under `sweep_<timestamp>` 
- **Tag**: A unique tag like `sweep_multirun_2025-11-04_14-30-52` for easy filtering
- **Additional tags**: Generation mode (e.g., `unconditional`) is automatically added

This makes it easy to view all related runs together in wandb by filtering by the shared tag or group.
