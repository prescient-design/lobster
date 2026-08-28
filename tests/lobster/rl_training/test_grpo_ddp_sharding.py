"""Tests for the single-node multi-GPU ``accum_targets`` sharding logic.

The distributed speedup interprets ``accum_targets`` as a GLOBAL per-step count and slices it
round-robin across ranks (``specs_all[rank::world_size]``). These are pure-logic checks — no
process group, no model — verifying that the slicing is a correct, non-overlapping partition and
that ``world_size == 1`` reproduces the single-GPU list byte-for-byte.

See :mod:`lobster.rl_training._leflur_grpo_trainer` ``train()`` (the ``specs_all`` / ``specs`` lines).
"""

import random

from lobster.rl_training._leflur_grpo_trainer import GRPOTrainerConfig, LeFlurGRPOTrainer, TargetSpec


def _spec(target_id: str) -> TargetSpec:
    return TargetSpec(target_id=target_id, antigen_pdb="x.pdb", target_chain="A", binder_length=80)


def _bare_trainer(config: GRPOTrainerConfig, targets: list[TargetSpec]) -> LeFlurGRPOTrainer:
    """Trainer shell without the heavy ``__init__`` (no model load)."""
    t = object.__new__(LeFlurGRPOTrainer)
    t.config = config
    t.targets = targets
    t._rng = random.Random(config.seed)
    t._sched_rng = random.Random(config.seed + 1000)
    return t


def _draw_all(t: LeFlurGRPOTrainer, sched, accum_targets: int) -> list[str]:
    """Every rank draws the FULL global batch each step (keeps the pointer in lockstep)."""
    return [next(sched).target_id for _ in range(max(1, accum_targets))]


def test_ws1_shard_is_identical_to_full_batch():
    """world_size == 1 -> specs_all[0::1] == specs_all (byte-identical single-GPU path)."""
    targets = [_spec(f"t{i}") for i in range(7)]
    cfg = GRPOTrainerConfig(shuffle_targets=False, accum_targets=10)
    t = _bare_trainer(cfg, targets)
    sched = t._iter_targets()

    specs_all = _draw_all(t, sched, cfg.accum_targets)
    world_size, rank = 1, 0
    shard = specs_all[rank::world_size]

    assert shard == specs_all


def test_shards_are_a_disjoint_cover():
    """Across ranks the shards partition specs_all with no overlap and no drops."""
    targets = [_spec(f"t{i}") for i in range(7)]
    cfg = GRPOTrainerConfig(shuffle_targets=False, accum_targets=10)
    t = _bare_trainer(cfg, targets)
    sched = t._iter_targets()

    specs_all = _draw_all(t, sched, cfg.accum_targets)

    for world_size in (2, 3, 4, 8):
        shards = [specs_all[r::world_size] for r in range(world_size)]
        # Reassembling by interleaving reproduces the original order exactly.
        rejoined: list[str] = []
        for i in range(len(specs_all)):
            rejoined.append(shards[i % world_size][i // world_size])
        assert rejoined == specs_all
        # Union covers every index once (multiset equality) and no pair overlaps positionally.
        flat = [x for s in shards for x in s]
        assert sorted(flat) == sorted(specs_all)
        assert sum(len(s) for s in shards) == len(specs_all)


def test_shard_sizes_are_balanced_for_uneven_split():
    """accum_targets not divisible by world_size -> sizes differ by at most 1 (round-robin)."""
    targets = [_spec(f"t{i}") for i in range(5)]
    cfg = GRPOTrainerConfig(shuffle_targets=False, accum_targets=5)  # 5 targets over 2 ranks -> 3 / 2
    t = _bare_trainer(cfg, targets)
    sched = t._iter_targets()

    specs_all = _draw_all(t, sched, cfg.accum_targets)
    world_size = 2
    sizes = [len(specs_all[r::world_size]) for r in range(world_size)]

    assert sizes == [3, 2]
    assert max(sizes) - min(sizes) <= 1


def test_pointer_stays_in_lockstep_across_steps():
    """Every rank advances the scheduler by the full accum_targets each step, so all ranks
    observe the SAME global batch every step (the round-robin pointer never drifts)."""
    targets = [_spec(f"t{i}") for i in range(6)]
    cfg = GRPOTrainerConfig(shuffle_targets=False, accum_targets=4)

    # Two independent trainers with identical seeds stand in for two ranks drawing the same schedule.
    t_a = _bare_trainer(cfg, [_spec(s.target_id) for s in targets])
    t_b = _bare_trainer(cfg, [_spec(s.target_id) for s in targets])
    sched_a, sched_b = t_a._iter_targets(), t_b._iter_targets()

    for _ in range(5):  # five optimizer steps
        batch_a = _draw_all(t_a, sched_a, cfg.accum_targets)
        batch_b = _draw_all(t_b, sched_b, cfg.accum_targets)
        assert batch_a == batch_b  # identical global batch on every rank, every step
