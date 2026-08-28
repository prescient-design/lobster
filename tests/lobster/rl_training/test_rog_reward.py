"""Tests for the radius-of-gyration compactness shaping reward.

Two layers:

* pure-numpy unit tests of :func:`rog_compactness` / :func:`rog_compactness_reward` —
  the length-normalized compactness form, a compact globule scoring above an
  over-extended chain, saturation at ``rog_full`` (a denser blob does not score past
  1.0), the ``< 2``-residue guard, binder-only masking (the antigen is ignored), and
  the degenerate zero-Rg guard;
* trainer-wiring tests of ``_rog_terms_for_group`` — the term is weighted by ``w_rog``,
  ``rog/*`` diagnostics are averaged, and a shared ``gen_bb`` is reused without decoding.

The trainer tests live here (not in ``test_trainers.py``) so they avoid importing
``lobster.rl_training.trainers`` (which pulls in ``trl``, absent in this env).
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lobster.rl_training import LeFlurGRPOTrainer
from lobster.rl_training.rewards import rog_compactness, rog_compactness_reward


# ------------------------------------------------------------------ fixtures
def _globule(n: int, radius: float | None = None, seed: int = 0) -> np.ndarray:
    """``n`` Cα points spread roughly uniformly through a constant-density ball.

    A real globular fold packs at constant density, so its enclosing radius grows as
    ``N^(1/3)``; by default the radius scales that way (density ~invariant in ``n``), which
    is what makes the compactness score length-normalized. Pass ``radius`` to override with
    a fixed radius (a denser or looser blob).
    """
    rng = np.random.default_rng(seed)
    r = radius if radius is not None else 1.6 * (n ** (1.0 / 3.0))
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)  # onto the unit sphere
    pts *= r * rng.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / 3.0)  # fill the ball
    return pts


def _extended(n: int, step: float = 3.8) -> np.ndarray:
    """``n`` Cα points on a straight line (maximally over-extended)."""
    ca = np.zeros((n, 3))
    ca[:, 0] = np.arange(n) * step
    return ca


def _bb_from_ca(ca: np.ndarray) -> np.ndarray:
    """Lift ``(N, 3)`` Cα coords to a ``(N, 3, 3)`` [N, CA, C] backbone (CA at index 1)."""
    bb = np.zeros((ca.shape[0], 3, 3))
    bb[:, 0] = ca + np.array([-1.46, 0.0, 0.0])  # N
    bb[:, 1] = ca  # CA
    bb[:, 2] = ca + np.array([0.98, 0.0, 0.0])  # C
    return bb


# --------------------------------------------------------------- unit: compactness
class TestRogCompactness:
    def test_compact_scores_above_extended(self):
        """A globule is more compact than a straight chain of the same length."""
        n = 40
        assert rog_compactness(_globule(n)) > rog_compactness(_extended(n))

    def test_length_normalized(self):
        """Compactness is ~scale-free in N for a fixed fold state (globule)."""
        c_small = rog_compactness(_globule(30, seed=1))
        c_large = rog_compactness(_globule(240, seed=2))
        # same fold state at 8x the residues -> compactness within a modest band
        assert c_small == pytest.approx(c_large, abs=0.25)

    def test_guard_too_few_points(self):
        assert rog_compactness(np.zeros((1, 3))) == 0.0
        assert rog_compactness(np.zeros((0, 3))) == 0.0

    def test_guard_zero_rg(self):
        """All points coincident -> Rg == 0 -> guarded to 0.0 (no divide-by-zero)."""
        assert rog_compactness(np.ones((5, 3))) == 0.0

    def test_r0_scales_linearly(self):
        ca = _globule(50)
        assert rog_compactness(ca, r0=4.4) == pytest.approx(2.0 * rog_compactness(ca, r0=2.2))


# --------------------------------------------------------------- unit: reward
class TestRogCompactnessReward:
    def test_compact_rewarded_above_extended(self):
        n = 40
        valid = np.ones(n, dtype=bool)
        binder = np.ones(n, dtype=bool)
        t_glob, d_glob = rog_compactness_reward(_bb_from_ca(_globule(n)), valid, binder)
        t_ext, d_ext = rog_compactness_reward(_bb_from_ca(_extended(n)), valid, binder)
        assert t_glob > t_ext
        assert 0.0 <= t_ext <= t_glob <= 1.0
        assert d_glob["compactness"] > d_ext["compactness"]
        assert d_glob["n_res"] == n

    def test_saturates_at_one(self):
        """A very dense blob saturates the clip at 1.0 (no reward for over-collapsing)."""
        ca = _globule(60, radius=1.0)  # tightly packed
        term, diag = rog_compactness_reward(_bb_from_ca(ca), np.ones(60, bool), np.ones(60, bool))
        assert term == pytest.approx(1.0)
        assert diag["compactness"] >= 0.76  # exceeds the native rog_full target

    def test_binder_only_masking(self):
        """Antigen Cα (masked out) never enter the compactness; only binder residues do."""
        binder_ca = _globule(20, seed=3)
        antigen_ca = _extended(20) + np.array([1000.0, 0.0, 0.0])  # far away, would wreck Rg
        ca_full = np.concatenate([antigen_ca, binder_ca], axis=0)
        bb = _bb_from_ca(ca_full)
        valid = np.ones(40, dtype=bool)
        binder = np.array([False] * 20 + [True] * 20)

        term, diag = rog_compactness_reward(bb, valid, binder)
        # identical to scoring the binder alone
        term_alone, diag_alone = rog_compactness_reward(_bb_from_ca(binder_ca), np.ones(20, bool), np.ones(20, bool))
        assert diag["n_res"] == 20
        assert term == pytest.approx(term_alone)
        assert diag["rg"] == pytest.approx(diag_alone["rg"])

    def test_invalid_positions_excluded(self):
        """valid_mask=False positions are dropped even when binder_mask=True."""
        ca = _globule(30, seed=4)
        bb = _bb_from_ca(ca)
        valid = np.ones(30, dtype=bool)
        valid[25:] = False  # last 5 padded
        binder = np.ones(30, dtype=bool)
        _, diag = rog_compactness_reward(bb, valid, binder)
        assert diag["n_res"] == 25

    def test_guard_too_few_binder_residues(self):
        bb = _bb_from_ca(_globule(10))
        valid = np.ones(10, dtype=bool)
        binder = np.zeros(10, dtype=bool)
        binder[0] = True  # only 1 binder residue
        term, diag = rog_compactness_reward(bb, valid, binder)
        assert term == 0.0
        assert diag == {"compactness": 0.0, "rg": 0.0, "n_res": 1}

    def test_rog_full_controls_saturation_point(self):
        """A lower rog_full grants full credit at a lower compactness."""
        ca = _globule(40, radius=8.0)  # moderately extended
        bb, valid, binder = _bb_from_ca(ca), np.ones(40, bool), np.ones(40, bool)
        t_strict, _ = rog_compactness_reward(bb, valid, binder, rog_full=0.76)
        t_loose, _ = rog_compactness_reward(bb, valid, binder, rog_full=0.40)
        assert t_loose >= t_strict


# -------------------------------------------------------------------- shim identity
def test_shim_is_same_module():
    """The old ``lobster`` import path is the *same* object as the plm_design_rl module."""
    import plm_design_rl.rewards._rog_reward as new
    import lobster.rl_training.rewards._rog_reward as old

    assert old is new
    assert old.rog_compactness_reward is rog_compactness_reward
    assert old.rog_compactness is rog_compactness


# ---------------------------------------------------------- trainer wiring
def _rog_cfg(**overrides):
    cfg = dict(w_rog=1.0, rog_r0=2.2, rog_full=0.76)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestRogTermsForGroup:
    def test_weights_and_aggregates(self, monkeypatch):
        """Term is weighted by w_rog and rog/* diagnostics are averaged."""
        import lobster.rl_training.rewards as _rewards

        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _rog_cfg(w_rog=2.0)
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 5, 3, 3))
        comp = {"mask": torch.ones(1, 5), "binder_positions": torch.ones(5)}

        diags = iter(
            [
                (1.00, {"compactness": 0.80, "rg": 9.0, "n_res": 5}),  # saturated
                (0.50, {"compactness": 0.38, "rg": 18.0, "n_res": 5}),
                (0.25, {"compactness": 0.19, "rg": 36.0, "n_res": 5}),
            ]
        )
        monkeypatch.setattr(_rewards, "rog_compactness_reward", lambda *a, **k: next(diags))

        weighted, metrics = trainer._rog_terms_for_group(trajectory={}, comp=comp)

        assert weighted == pytest.approx([2.0, 1.0, 0.5])  # 2.0 * term
        assert metrics["reward/rog_term_mean"] == pytest.approx((2.0 + 1.0 + 0.5) / 3)
        assert metrics["rog/compactness_mean"] == pytest.approx((0.80 + 0.38 + 0.19) / 3)
        assert metrics["rog/rg_mean"] == pytest.approx((9.0 + 18.0 + 36.0) / 3)
        assert metrics["rog/n_res_mean"] == pytest.approx(5.0)
        # only the first design's compactness (0.80) clears rog_full=0.76
        assert metrics["rog/frac_saturated"] == pytest.approx(1 / 3)

    def test_uses_shared_gen_bb_without_decoding(self):
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _rog_cfg()
        trainer._decode_backbone_coords = Mock(side_effect=AssertionError("must not decode"))
        comp = {"mask": torch.ones(1, 40), "binder_positions": torch.ones(40)}
        gen_bb = np.stack([_bb_from_ca(_globule(40, seed=5)), _bb_from_ca(_extended(40))])  # (2,40,3,3)

        weighted, metrics = trainer._rog_terms_for_group(trajectory={}, comp=comp, gen_bb=gen_bb)

        assert len(weighted) == 2
        assert weighted[0] > weighted[1]  # globule beats the straight chain
        assert all(0.0 <= w <= 1.0 for w in weighted)
        trainer._decode_backbone_coords.assert_not_called()
