# Distributed Generation with Hydra Submitit

## Basic Usage

Run generation across multiple GPUs using Hydra's submitit launcher:

```bash
uv run python -m lobster.cmdline.generate \
  --multirun \
  experiment=generate_unconditional \
  hydra/launcher=submitit_slurm \
  generation.num_samples=100 \
  generation.length=[500] \
  generation.nsteps=1000 \
  seed=12345
```
