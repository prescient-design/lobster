"""Tests for multi-target gradient accumulation in ``LeFlurGRPOTrainer``.

These exercise the pure orchestration logic added for the multi-target stabilization
arm (``accum_targets`` / ``shuffle_targets``) without constructing a real policy model:

* ``_iter_targets`` — round-robin schedule (byte-identical to ``step % n`` when
  shuffling is off) and per-epoch reshuffling that still covers every target.
* ``_merge_step_metrics`` — single-packet passthrough vs multi-packet averaging.
* ``_ppo_update`` — the accumulation math: averaging N identical targets reproduces the
  single-target gradient, flat groups are dropped, and an all-flat batch is a no-op.
"""

import random

import pytest
import torch

from lobster.rl_training._leflur_grpo_trainer import GRPOTrainerConfig, LeFlurGRPOTrainer, TargetSpec


def _spec(target_id: str) -> TargetSpec:
    return TargetSpec(target_id=target_id, antigen_pdb="x.pdb", target_chain="A", binder_length=80)


def _bare_trainer(config: GRPOTrainerConfig, targets: list[TargetSpec]) -> LeFlurGRPOTrainer:
    """Construct a trainer shell without running the heavy ``__init__`` (no model load)."""
    t = object.__new__(LeFlurGRPOTrainer)
    t.config = config
    t.targets = targets
    t._rng = random.Random(config.seed)
    t._sched_rng = random.Random(config.seed + 1000)
    return t


# --------------------------------------------------------------------- schedule
def test_iter_targets_roundrobin_matches_step_modulo():
    """shuffle_targets=False yields the legacy ``targets[step % n]`` order exactly."""
    targets = [_spec(f"t{i}") for i in range(5)]
    cfg = GRPOTrainerConfig(shuffle_targets=False)
    t = _bare_trainer(cfg, targets)

    sched = t._iter_targets()
    drawn = [next(sched).target_id for _ in range(13)]
    expected = [targets[i % 5].target_id for i in range(13)]
    assert drawn == expected


def test_iter_targets_shuffle_covers_every_target_per_epoch():
    """shuffle_targets=True reshuffles per epoch but still covers all targets once each."""
    targets = [_spec(f"t{i}") for i in range(6)]
    cfg = GRPOTrainerConfig(shuffle_targets=True, seed=0)
    t = _bare_trainer(cfg, targets)

    sched = t._iter_targets()
    epoch1 = [next(sched).target_id for _ in range(6)]
    epoch2 = [next(sched).target_id for _ in range(6)]

    # Each epoch is a permutation of all targets (a full cover, no repeats/drops).
    assert sorted(epoch1) == [s.target_id for s in targets]
    assert sorted(epoch2) == [s.target_id for s in targets]
    # And the order is actually shuffled (astronomically unlikely to match round-robin twice).
    assert not (epoch1 == [s.target_id for s in targets] and epoch2 == [s.target_id for s in targets])


def test_iter_targets_does_not_touch_main_rng_when_shuffle_off():
    """With shuffling off the scheduler must never draw from ``self._rng``."""
    targets = [_spec(f"t{i}") for i in range(4)]
    cfg = GRPOTrainerConfig(shuffle_targets=False)
    t = _bare_trainer(cfg, targets)

    before = t._rng.getstate()
    sched = t._iter_targets()
    for _ in range(10):
        next(sched)
    assert t._rng.getstate() == before  # step-subset / binder-length draw order preserved


# ----------------------------------------------------------------- merge metrics
def test_merge_single_packet_passthrough():
    """One packet: metrics pass through unchanged plus the update dict + string target."""
    packet = {
        "spec": _spec("PD1"),
        "flat": False,
        "metrics": {"reward/mean": 0.7, "advantage/std": 0.1},
    }
    update = {"ppo/pg_loss": -0.3, "ppo/ratio_mean": 1.0}
    merged = LeFlurGRPOTrainer._merge_step_metrics([packet], update)

    assert merged["reward/mean"] == 0.7
    assert merged["advantage/std"] == 0.1
    assert merged["ppo/pg_loss"] == -0.3
    assert merged["target"] == "PD1"
    assert "update/batch_targets" not in merged  # single-packet path stays legacy-shaped


def test_merge_multi_packet_averages_and_joins_targets():
    """Multiple packets: numeric metrics averaged, target ids joined, batch size logged."""
    p1 = {"spec": _spec("A"), "flat": False, "metrics": {"reward/mean": 0.4, "conf/pass_rate": 0.2}}
    p2 = {"spec": _spec("B"), "flat": False, "metrics": {"reward/mean": 0.6, "conf/pass_rate": 0.4}}
    update = {"ppo/pg_loss": -0.1, "update/n_targets": 2.0}
    merged = LeFlurGRPOTrainer._merge_step_metrics([p1, p2], update)

    assert merged["reward/mean"] == pytest.approx(0.5)
    assert merged["conf/pass_rate"] == pytest.approx(0.3)
    assert merged["ppo/pg_loss"] == pytest.approx(-0.1)
    assert merged["target"] == "A,B"
    assert merged["update/batch_targets"] == 2.0


# ------------------------------------------------------------------- ppo update
class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


class _FakeModel:
    """Minimal policy stub: log-prob is an affine function of the single param ``w``."""

    def __init__(self, encoder: _FakeEncoder) -> None:
        self.encoder = encoder

    def logprob_over_trajectory(self, trajectory, tracks, step_indices, grad_checkpoint=False, per_position_tracks=()):
        # new_lp = base + w * scale (PER-DESIGN scale so the ratio varies across the group;
        # a uniform scale would make the first-step gradient of mean-centered advantages
        # exactly zero). At w=0, new_lp == base == old_lp -> ratio 1.
        return trajectory["base_lp"] + self.encoder.w * trajectory["scale"]


def _make_ppo_trainer(mu: int = 1) -> LeFlurGRPOTrainer:
    cfg = GRPOTrainerConfig(
        mu=mu,
        beta=0.0,
        eps_clip=0.2,
        grad_clip=0.0,  # disable clipping so grads compare exactly
        steps_per_update=4,  # >= n_steps below -> subset = all steps, no RNG draw
        lr=0.1,
    )
    t = object.__new__(LeFlurGRPOTrainer)
    t.config = cfg
    enc = _FakeEncoder()
    t.model = _FakeModel(enc)
    t.optimizer = torch.optim.SGD(enc.parameters(), lr=cfg.lr)
    t._rng = random.Random(0)
    return t


def _packet(adv: list[float], base: list[float], scale: list[float], flat: bool = False) -> dict:
    advantages = torch.tensor(adv, dtype=torch.float32)
    base_lp = torch.tensor(base, dtype=torch.float32)
    trajectory = {"base_lp": base_lp, "scale": torch.tensor(scale, dtype=torch.float32), "steps": [0]}
    packet = {"spec": _spec("t"), "metrics": {}, "flat": flat}
    if not flat:
        packet.update(
            trajectory=trajectory,
            advantages=advantages,
            old_lp_per_step=base_lp.unsqueeze(0),  # (n_steps=1, G); sum over subset == base_lp
            n_steps=1,
        )
    return packet


def test_ppo_update_two_identical_targets_equals_single():
    """Averaging two identical targets yields the same optimizer step as one target."""
    adv, base, scale = [1.0, -1.0, 0.5, -0.5], [0.0, 0.0, 0.0, 0.0], [0.3, 0.1, 0.2, 0.4]

    t_single = _make_ppo_trainer()
    t_single._ppo_update([_packet(adv, base, scale)])
    w_single = float(t_single.model.encoder.w.detach())

    t_double = _make_ppo_trainer()
    t_double._ppo_update([_packet(adv, base, scale), _packet(adv, base, scale)])
    w_double = float(t_double.model.encoder.w.detach())

    assert w_single != 0.0  # the update actually moved the parameter
    assert abs(w_single - w_double) < 1e-6


def test_ppo_update_drops_flat_packets():
    """A flat packet contributes nothing; batch [live, flat] == batch [live]."""
    adv, base, scale = [1.0, -1.0, 0.5, -0.5], [0.0, 0.0, 0.0, 0.0], [0.3, 0.1, 0.2, 0.4]

    t_live = _make_ppo_trainer()
    t_live._ppo_update([_packet(adv, base, scale)])
    w_live = float(t_live.model.encoder.w.detach())

    t_mixed = _make_ppo_trainer()
    metrics = t_mixed._ppo_update([_packet(adv, base, scale), _packet(adv, base, scale, flat=True)])
    w_mixed = float(t_mixed.model.encoder.w.detach())

    assert abs(w_live - w_mixed) < 1e-6
    assert metrics["update/n_targets"] == 1.0


def test_ppo_update_all_flat_is_noop():
    """An all-flat batch returns no metrics and never steps the optimizer."""
    adv, base, scale = [1.0, -1.0], [0.0, 0.0], [0.3, 0.1]
    t = _make_ppo_trainer()
    metrics = t._ppo_update([_packet(adv, base, scale, flat=True)])
    assert metrics == {}
    assert float(t.model.encoder.w.detach()) == 0.0
