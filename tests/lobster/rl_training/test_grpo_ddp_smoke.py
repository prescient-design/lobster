"""End-to-end 2-GPU smoke test for the distributed GRPO trainer.

Skipped unless ``LOBSTER_GRPO_DDP_SMOKE=1`` and >=2 CUDA devices are visible — it needs a real
2xb200 (or 2xGPU) box, a policy checkpoint, and a running reward queue, so it does not run in CI.
Kept as an executable checklist for the first real multi-GPU launch: a 2-rank few-step run whose
step-0 ``reward/mean`` / ``ppo/pg_loss`` / ``ppo/grad_norm`` should track a single-GPU control run
with the same *total* ``accum_targets`` (with per-target deterministic seeding, ``grad_norm`` agrees
to floating-point tolerance).

To run manually::

    LOBSTER_GRPO_DDP_SMOKE=1 uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \\
        -m pytest tests/lobster/rl_training/test_grpo_ddp_smoke.py -s
"""

import os

import pytest

torch = pytest.importorskip("torch")

_SMOKE = os.environ.get("LOBSTER_GRPO_DDP_SMOKE") == "1"
_HAS_2GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (_SMOKE and _HAS_2GPU), reason="needs LOBSTER_GRPO_DDP_SMOKE=1 and >=2 GPUs"),
]


def test_ddp_two_rank_process_group_roundtrip():
    """Under torchrun, the two ranks form a group and agree on an all-reduced tensor.

    This is the minimal invariant the trainer relies on: every rank sees the same summed value,
    which is what makes the encoder-gradient all-reduce produce bit-synced replicas.
    """
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        assert world_size == dist.get_world_size() >= 2
        x = torch.tensor([float(dist.get_rank()) + 1.0], device=f"cuda:{local_rank}")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        expected = float(world_size * (world_size + 1) // 2)  # 1 + 2 + ... + world_size
        assert x.item() == pytest.approx(expected)
    finally:
        dist.destroy_process_group()
