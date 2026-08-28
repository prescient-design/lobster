"""Numerical-equivalence test for the distributed encoder-gradient all-reduce.

``LeFlurGRPOTrainer._allreduce_and_scale_encoder_grads`` sums each rank's *unscaled* accumulated
encoder gradients, sums the per-rank live-target counts, and divides the reduced gradient by that
global count. The result (identical on every rank) must equal the single-GPU accumulation

    (Σ_ranks Σ_targets grad_i) / (Σ_ranks n_live_local)

to floating-point tolerance, including when the shards are UNEVEN (a rank with more live targets
than another) and when a rank's shard is entirely flat (``n_live_local == 0`` -> ``None`` grads
coalesced to zeros).

Runs on CPU via the gloo backend and ``torch.multiprocessing.spawn`` so it is CI-friendly (no GPU,
no NCCL). See :mod:`lobster.rl_training._leflur_grpo_trainer`.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lobster.rl_training._leflur_grpo_trainer import LeFlurGRPOTrainer


class _Encoder(torch.nn.Module):
    """Two params of different shapes, to exercise the flatten/unflatten path."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(3, 4))
        self.b = torch.nn.Parameter(torch.zeros(5))


class _Model:
    def __init__(self, encoder: _Encoder) -> None:
        self.encoder = encoder


def _bare_trainer_with_grads(grads: list[torch.Tensor] | None) -> LeFlurGRPOTrainer:
    """Trainer shell carrying a fresh encoder whose ``.grad`` is preset to ``grads`` (or None)."""
    t = object.__new__(LeFlurGRPOTrainer)
    t.model = _Model(_Encoder())
    t.device = torch.device("cpu")
    params = list(t.model.encoder.parameters())
    if grads is None:
        for p in params:
            p.grad = None
    else:
        for p, g in zip(params, grads):
            p.grad = g.clone()
    return t


# Per-rank preset gradients. Rank 0: 3 live targets; rank 1: 2 live targets (UNEVEN). Values are
# the already-summed local gradients (what accumulates after N unscaled backward()s).
_RANK_GRADS = {
    0: ([torch.full((3, 4), 6.0), torch.full((5,), 3.0)], 3),  # sum over 3 targets
    1: ([torch.full((3, 4), 4.0), torch.full((5,), 8.0)], 2),  # sum over 2 targets
}


def _expected_mean() -> list[torch.Tensor]:
    """Single-process reference: (Σ grads) / (Σ n_live)."""
    total_n = sum(n for _, n in _RANK_GRADS.values())
    acc_a = sum(g[0] for g, _ in _RANK_GRADS.values())
    acc_b = sum(g[1] for g, _ in _RANK_GRADS.values())
    return [acc_a / total_n, acc_b / total_n]


def _worker(rank: int, world_size: int, tmp_file: str, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_file}",
        world_size=world_size,
        rank=rank,
    )
    try:
        grads, n_live = _RANK_GRADS[rank]
        t = _bare_trainer_with_grads(grads)
        global_n = t._allreduce_and_scale_encoder_grads(n_live)

        assert global_n == sum(n for _, n in _RANK_GRADS.values())
        params = list(t.model.encoder.parameters())
        for p, ref in zip(params, _expected_mean()):
            torch.testing.assert_close(p.grad, ref, atol=1e-6, rtol=0)

        # Every rank must end with the identical reduced gradient (replicas stay bit-synced).
        torch.save([p.grad for p in params], os.path.join(out_dir, f"grad_rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _worker_all_flat(rank: int, world_size: int, tmp_file: str, out_dir: str) -> None:
    """Every rank has an all-flat shard (n_live_local == 0, grads None) -> global skip."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_file}",
        world_size=world_size,
        rank=rank,
    )
    try:
        t = _bare_trainer_with_grads(None)
        global_n = t._allreduce_and_scale_encoder_grads(0)
        assert global_n == 0  # caller skips the optimizer step identically on all ranks
        with open(os.path.join(out_dir, f"flat_rank{rank}.ok"), "w") as fh:
            fh.write("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
def test_allreduce_scale_matches_single_process_uneven(world_size, tmp_path):
    """Uneven 3-vs-2 shard: reduced+scaled grad == single-process Σgrad/Σn_live on both ranks."""
    tmp_file = str(tmp_path / "pg_init")
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)

    mp.spawn(_worker, args=(world_size, tmp_file, out_dir), nprocs=world_size, join=True)

    ref = _expected_mean()
    grads_by_rank = [torch.load(os.path.join(out_dir, f"grad_rank{r}.pt")) for r in range(world_size)]
    for rank_grads in grads_by_rank:
        for got, exp in zip(rank_grads, ref):
            torch.testing.assert_close(got, exp, atol=1e-6, rtol=0)
    # Ranks agree with each other, too.
    for got0, got1 in zip(grads_by_rank[0], grads_by_rank[1]):
        torch.testing.assert_close(got0, got1, atol=0, rtol=0)


@pytest.mark.parametrize("world_size", [2])
def test_allreduce_all_flat_returns_zero(world_size, tmp_path):
    """All ranks flat -> global_n_live == 0 so the caller skips the step (no deadlock)."""
    tmp_file = str(tmp_path / "pg_init_flat")
    out_dir = str(tmp_path / "out_flat")
    os.makedirs(out_dir, exist_ok=True)

    mp.spawn(_worker_all_flat, args=(world_size, tmp_file, out_dir), nprocs=world_size, join=True)

    for r in range(world_size):
        assert os.path.exists(os.path.join(out_dir, f"flat_rank{r}.ok"))
