"""Tests for the whole-binder all-atom side-chain steric-clash (SC-clash) reward.

Three layers, mirroring ``test_shape_reward.py`` / ``test_clash_reward.py``:

* the pure-numpy reward math (:mod:`lobster.rl_training.rewards._sc_clash_reward`):
  the ``binder_clash_terms`` aggregator, the ``sc_clash_reward`` mapper, and the
  atom14→cloud helpers ``cloud_from_atom14`` / ``_ca_by_residue``,
* the trainer wiring — ``LeFlurGRPOTrainer._repack_terms_for_group`` (the shared
  LigandMPNN-repack pool round-trip that serves SC / clash / AAR from one packing), and
* the ``_compute_rewards`` routing gate: SC-only stays byte-identical (routes through the
  unchanged ``_shape_terms_for_group``); any set that adds clash/aar routes through
  ``_repack_terms_for_group``; all-zero weights are fully inert.

The trainer tests live here (not in ``test_trainers.py``) so they avoid importing
``lobster.rl_training.trainers`` (which pulls in ``trl``, absent in this env).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lobster.rl_training import LeFlurGRPOTrainer
from lobster.rl_training.rewards import sc_clash_reward
from lobster.rl_training.rewards._sc_clash_reward import (
    binder_clash_terms,
    cloud_from_atom14,
)
from lobster.rl_training.rewards._sc_clash_reward import _ca_by_residue


# --------------------------------------------------------------- atom14 helpers
def _atom14_block(coords_by_res: list[np.ndarray], s_ints: list[int]):
    """``list[(n_atoms, 3)]`` -> ``(L, 14, 3)`` X14, ``(L, 14)`` X_m, ``(L,)`` S."""
    L = len(coords_by_res)
    X14 = np.zeros((L, 14, 3), dtype=np.float64)
    X_m = np.zeros((L, 14), dtype=np.float64)
    for i, c in enumerate(coords_by_res):
        na = c.shape[0]
        X14[i, :na] = c
        X_m[i, :na] = 1.0
    return X14, X_m, np.asarray(s_ints, dtype=np.int64)


def _ala(ca_xyz: np.ndarray) -> np.ndarray:
    """One ALA residue (N, CA, C, O, CB) placed compactly around ``ca_xyz`` (CA at idx 1)."""
    off = np.array([[1.4, 0, 0], [0, 0, 0], [-0.7, 1.2, 0], [-0.7, 2.2, 0.3], [0.2, -0.9, 1.0]])
    return ca_xyz[None, :] + off


# --------------------------------------------------------------- reward mapper
class TestSCClashReward:
    def test_none_is_floor(self):
        assert sc_clash_reward(None) == 0.0

    def test_clashfree_is_one(self):
        assert sc_clash_reward({"E_clash_total": 0.0}) == pytest.approx(1.0)

    def test_monotone_decreasing_bounded(self):
        r_small = sc_clash_reward({"E_clash_total": 15.0})
        r_big = sc_clash_reward({"E_clash_total": 300.0})
        assert 0.0 < r_big < r_small < 1.0

    def test_fallback_to_binder_plus_iface(self):
        assert sc_clash_reward({"E_clash_binder": 10.0, "E_clash_iface": 5.0}) == pytest.approx(
            sc_clash_reward({"E_clash_total": 15.0})
        )

    def test_scale_arg(self):
        """A larger saturation scale gives a higher reward for the same energy."""
        e = {"E_clash_total": 100.0}
        assert sc_clash_reward(e, sc_clash_scale=300.0) > sc_clash_reward(e, sc_clash_scale=50.0)

    def test_density_uses_per_residue_energy(self):
        """Density mode maps ``e_binder_norm + E_iface_res/n_iface_res`` through exp(-E/scale)."""
        import numpy as np

        res = {"e_clash_binder_norm": 4.0, "E_clash_iface_res": 80.0, "n_iface_res": 10}
        # density = 4.0 + 80/10 = 12.0
        assert sc_clash_reward(res, density=True, sc_clash_density_scale=12.0) == pytest.approx(float(np.exp(-1.0)))

    def test_density_empty_interface_is_clashfree(self):
        """No interface residues => iface term 0; a clash-free binder => density 0 => reward 1."""
        assert sc_clash_reward(
            {"e_clash_binder_norm": 0.0, "E_clash_iface_res": 0.0, "n_iface_res": 0}, density=True
        ) == pytest.approx(1.0)

    def test_density_is_retraction_resistant(self):
        """DENSITY is flat under pure retraction (same per-residue clash, fewer iface res),
        whereas ABSOLUTE ``E_clash_total`` is *lowered* by retraction (rewarding it)."""
        # Baseline: 30 interface residues, per-iface-res clash 7.0; whole-binder norm 4.0.
        base = {
            "e_clash_binder_norm": 4.0,
            "E_clash_iface_res": 7.0 * 30,
            "n_iface_res": 30,
            "E_clash_binder": 400.0,
            "E_clash_iface": 7.0 * 30,
            "E_clash_total": 400.0 + 7.0 * 30,
        }
        # Retracted: same per-residue clash but only 8 interface residues retained.
        retract = {
            "e_clash_binder_norm": 4.0,
            "E_clash_iface_res": 7.0 * 8,
            "n_iface_res": 8,
            "E_clash_binder": 400.0,
            "E_clash_iface": 7.0 * 8,
            "E_clash_total": 400.0 + 7.0 * 8,
        }
        # Density: identical (numerator and denominator shrink together).
        assert sc_clash_reward(base, density=True) == pytest.approx(sc_clash_reward(retract, density=True))
        # Absolute: retraction lowers E_total, so it is *rewarded* (higher term) — the gaming mode.
        assert sc_clash_reward(retract) > sc_clash_reward(base)

    def test_density_nan_guard_and_bounds(self):
        assert sc_clash_reward(None, density=True) == 0.0
        assert (
            sc_clash_reward(
                {"e_clash_binder_norm": float("nan"), "E_clash_iface_res": 1.0, "n_iface_res": 5},
                density=True,
            )
            == 0.0
        )
        r = sc_clash_reward({"e_clash_binder_norm": 100.0, "E_clash_iface_res": 500.0, "n_iface_res": 5}, density=True)
        assert 0.0 <= r <= 1.0


# --------------------------------------------------------------- atom14 -> cloud
class TestCloudFromAtom14:
    def test_shapes_elements_residx(self):
        res = [_ala(np.array([0.0, 0, 0])), _ala(np.array([10.0, 0, 0]))]
        X14, X_m, S = _atom14_block(res, [0, 0])
        xyz, elem, ridx = cloud_from_atom14(X14, X_m, S)
        assert xyz.shape == (10, 3)
        assert elem == ["N", "C", "C", "O", "C"] * 2
        assert list(ridx) == [0] * 5 + [1] * 5

    def test_empty(self):
        xyz, elem, ridx = cloud_from_atom14(np.zeros((0, 14, 3)), np.zeros((0, 14)), np.zeros(0))
        assert xyz.shape == (0, 3) and elem == [] and ridx.shape == (0,)


class TestCaByResidue:
    def test_ca_at_atom14_index_1(self):
        res = [_ala(np.array([0.0, 0, 0])), _ala(np.array([10.0, 0, 0]))]
        X14, X_m, _ = _atom14_block(res, [0, 0])
        ca = _ca_by_residue(X14, X_m)
        assert ca.shape == (2, 3)
        assert np.allclose(ca[0], [0, 0, 0]) and np.allclose(ca[1], [10, 0, 0])

    def test_empty(self):
        assert _ca_by_residue(np.zeros((0, 14, 3)), np.zeros((0, 14))).shape == (0, 3)


# --------------------------------------------------------------- clash aggregator
class TestBinderClashTerms:
    def _far_antigen(self):
        return _atom14_block([_ala(np.array([0.0, 0, 200.0])), _ala(np.array([8.0, 0, 200.0]))], [0, 0])

    def test_clashfree_far(self):
        bd = [_ala(np.array([x, 0.0, 0.0])) for x in (0, 8, 16, 24)]
        bd_X14, bd_Xm, bd_S = _atom14_block(bd, [0, 0, 0, 0])
        ag_X14, ag_Xm, ag_S = self._far_antigen()
        t = binder_clash_terms(bd_X14, bd_Xm, bd_S, ag_X14, ag_Xm, ag_S)
        assert t["E_clash_binder"] < 1e-2 and t["E_clash_iface"] < 1e-2
        assert t["term"] == pytest.approx(1.0, abs=1e-3)
        assert t["n_iface_res"] == 0 and t["E_clash_iface_res"] == 0.0
        assert t["n_bd_atoms"] == 20 and t["n_ag_atoms"] == 10

    def test_selfclash_raises_energy_drops_term(self):
        far = [_ala(np.array([x, 0.0, 0.0])) for x in (0, 8, 16, 24)]
        far_X14, far_Xm, far_S = _atom14_block(far, [0, 0, 0, 0])
        ag_X14, ag_Xm, ag_S = self._far_antigen()
        t_far = binder_clash_terms(far_X14, far_Xm, far_S, ag_X14, ag_Xm, ag_S)

        fold = [_ala(np.array([x, 0.0, 0.0])) for x in (0, 8, 16)]
        fold.append(_ala(np.array([0.3, 0.0, 0.0])))  # residue 3 slammed onto residue 0
        f_X14, f_Xm, f_S = _atom14_block(fold, [0, 0, 0, 0])
        t_self = binder_clash_terms(f_X14, f_Xm, f_S, ag_X14, ag_Xm, ag_S)
        assert t_self["E_clash_binder"] > t_far["E_clash_binder"] + 1.0
        assert t_self["term"] < t_far["term"]
        assert t_self["n_clash_atoms_binder"] > 0

    def test_interface_clash_and_diagnostic(self):
        bd = [_ala(np.array([x, 0.0, 0.0])) for x in (0, 8, 16, 24)]
        bd_X14, bd_Xm, bd_S = _atom14_block(bd, [0, 0, 0, 0])
        ag_X14, ag_Xm, ag_S = self._far_antigen()
        t_far = binder_clash_terms(bd_X14, bd_Xm, bd_S, ag_X14, ag_Xm, ag_S)

        ag_near = [_ala(np.array([0.3, 0.0, 0.0])), _ala(np.array([8.0, 0.5, 0.0]))]
        n_X14, n_Xm, n_S = _atom14_block(ag_near, [0, 0])
        t_if = binder_clash_terms(bd_X14, bd_Xm, bd_S, n_X14, n_Xm, n_S)
        assert t_if["E_clash_iface"] > t_far["E_clash_iface"] + 1.0
        assert t_if["term"] < t_far["term"]
        assert t_if["n_iface_res"] > 0 and t_if["E_clash_iface_res"] > 0.0
        # diagnostic (interface-restricted) is a subset of the whole-binder iface energy
        assert t_if["E_clash_iface_res"] <= t_if["E_clash_iface"] + 1e-6

    def test_empty_binder_is_clashfree(self):
        ag_X14, ag_Xm, ag_S = self._far_antigen()
        t = binder_clash_terms(np.zeros((0, 14, 3)), np.zeros((0, 14)), np.zeros(0), ag_X14, ag_Xm, ag_S)
        assert t["E_clash_total"] == 0.0 and t["term"] == pytest.approx(1.0)
        assert t["n_iface_res"] == 0

    def test_total_is_binder_plus_iface(self):
        bd = [_ala(np.array([x, 0.0, 0.0])) for x in (0, 8, 16, 24)]
        bd_X14, bd_Xm, bd_S = _atom14_block(bd, [0, 0, 0, 0])
        ag_near = [_ala(np.array([0.3, 0.0, 0.0])), _ala(np.array([8.0, 0.5, 0.0]))]
        n_X14, n_Xm, n_S = _atom14_block(ag_near, [0, 0])
        t = binder_clash_terms(bd_X14, bd_Xm, bd_S, n_X14, n_Xm, n_S)
        assert t["E_clash_total"] == pytest.approx(t["E_clash_binder"] + t["E_clash_iface"])


# ------------------------------------------------------------- trainer wiring
def _fake_model(G: int, L: int):
    """A stand-in model exposing decode_endpoint_aa -> (G, L) AA ids."""
    m = Mock()
    m.decode_endpoint_aa = Mock(return_value=torch.zeros(G, L, dtype=torch.long))
    return m


def _mk_trainer(G: int, L: int, **cfg_overrides):
    trainer = object.__new__(LeFlurGRPOTrainer)
    trainer.device = "cpu"
    cfg = dict(w_shape=0.0, w_sc_clash=0.0, w_aar=0.0)
    cfg.update(cfg_overrides)
    trainer.config = SimpleNamespace(**cfg)
    trainer.model = _fake_model(G, L)
    return trainer


def _comp(L: int, binder_positions):
    return {"mask": torch.ones(1, L), "binder_positions": torch.tensor(binder_positions)}


class TestRepackTermsForGroup:
    """One shared round-trip, mapped through the three metric mappers per ``want``."""

    def test_clash_only(self):
        G, L = 3, 5
        trainer = _mk_trainer(G, L, w_sc_clash=2.0)
        gen_bb = np.zeros((G, L, 3, 3), dtype=np.float32)
        comp = _comp(L, [0, 0, 0, 1, 1])
        # nested clash results (whole-binder E_clash_total); one failed design -> floor 0
        results = [
            {
                "clash": {
                    "E_clash_total": 0.0,
                    "E_clash_binder": 0.0,
                    "E_clash_iface": 0.0,
                    "E_clash_iface_res": 0.0,
                    "n_iface_res": 0,
                }
            },
            {
                "clash": {
                    "E_clash_total": 150.0,
                    "E_clash_binder": 100.0,
                    "E_clash_iface": 50.0,
                    "E_clash_iface_res": 20.0,
                    "n_iface_res": 3,
                }
            },
            None,
        ]
        trainer._shape_client = Mock()
        captured = {}

        def _score(target_id, designs, want, return_seq=False):
            captured["want"] = want
            captured["designs"] = designs
            return results

        trainer._shape_client.score_group = _score
        shape_t, clash_t, aar_t, metrics, *_ = trainer._repack_terms_for_group(
            "t0", {}, comp, ["CC", "CC", "CC"], ("clash",), gen_bb=gen_bb
        )
        assert captured["want"] == ("clash",)
        # design split: 3 antigen residues, 2 binder residues
        d0 = captured["designs"][0]
        assert d0["ag_bb"].shape == (3, 3, 3) and d0["bd_bb"].shape == (2, 3, 3)
        assert shape_t == [0.0, 0.0, 0.0] and aar_t == [0.0, 0.0, 0.0]  # not requested
        assert clash_t[0] == pytest.approx(2.0 * 1.0)  # E=0 -> reward 1 * weight 2
        assert clash_t[1] == pytest.approx(2.0 * sc_clash_reward({"E_clash_total": 150.0}))
        assert clash_t[2] == 0.0  # failed design floored
        assert metrics["sc_clash/scored_frac"] == pytest.approx(2 / 3)
        assert metrics["sc_clash/e_total_mean"] == pytest.approx((0.0 + 150.0) / 2)
        assert "reward/sc_clash_term_mean" in metrics

    def test_aar_only(self):
        G, L = 2, 4
        trainer = _mk_trainer(G, L, w_aar=1.0)
        gen_bb = np.zeros((G, L, 3, 3), dtype=np.float32)
        comp = _comp(L, [0, 0, 1, 1])
        results = [
            {"aar": {"aar": 0.4, "aar_iface": 0.6, "c_mpnn": 0.3, "c_mpnn_iface": 0.5}},
            {
                "aar": {
                    "aar": float("nan"),
                    "aar_iface": float("nan"),
                    "c_mpnn": float("nan"),
                    "c_mpnn_iface": float("nan"),
                }
            },
        ]
        trainer._shape_client = Mock()
        trainer._shape_client.score_group = Mock(return_value=results)
        shape_t, clash_t, aar_t, metrics, *_ = trainer._repack_terms_for_group(
            "t0", {}, comp, ["CC", "CC"], ("aar",), gen_bb=gen_bb
        )
        assert shape_t == [0.0, 0.0] and clash_t == [0.0, 0.0]
        assert aar_t[0] == pytest.approx(1.0 * 0.4)
        assert aar_t[1] == 0.0  # nan aar -> floored
        assert metrics["aar/aar_mean"] == pytest.approx(0.4)  # nan excluded from mean
        assert metrics["aar/scored_frac"] == pytest.approx(1.0)  # both dicts present

    def test_combined_sc_clash_aar_single_round_trip(self):
        """All three metrics from ONE score_group call; SC flat, clash/aar nested."""
        G, L = 2, 4
        trainer = _mk_trainer(G, L, w_shape=1.0, w_sc_clash=1.0, w_aar=1.0)
        gen_bb = np.zeros((G, L, 3, 3), dtype=np.float32)
        comp = _comp(L, [0, 0, 1, 1])
        results = [
            {
                "term": 0.5,
                "sc": 0.5,
                "n_patch_a": 10,
                "n_patch_b": 8,  # SC flat at top
                "clash": {
                    "E_clash_total": 0.0,
                    "E_clash_binder": 0.0,
                    "E_clash_iface": 0.0,
                    "E_clash_iface_res": 0.0,
                    "n_iface_res": 0,
                },
                "aar": {"aar": 0.7, "aar_iface": 0.8, "c_mpnn": 0.4, "c_mpnn_iface": 0.6},
            },
            {
                "term": 0.2,
                "sc": 0.2,
                "n_patch_a": 12,
                "n_patch_b": 9,
                "clash": {
                    "E_clash_total": 150.0,
                    "E_clash_binder": 100.0,
                    "E_clash_iface": 50.0,
                    "E_clash_iface_res": 20.0,
                    "n_iface_res": 2,
                },
                "aar": {"aar": 0.3, "aar_iface": 0.4, "c_mpnn": 0.2, "c_mpnn_iface": 0.3},
            },
        ]
        calls = {"n": 0}

        def _score(target_id, designs, want, return_seq=False):
            calls["n"] += 1
            assert want == ("sc", "clash", "aar")  # order _compute_rewards builds
            return results

        trainer._shape_client = Mock()
        trainer._shape_client.score_group = _score
        shape_t, clash_t, aar_t, metrics, *_ = trainer._repack_terms_for_group(
            "t0", {}, comp, ["CC", "CC"], ("sc", "clash", "aar"), gen_bb=gen_bb
        )
        assert calls["n"] == 1  # packed ONCE for all three metrics
        assert shape_t == pytest.approx([0.5, 0.2])
        assert clash_t[0] == pytest.approx(1.0)  # E=0 -> reward 1
        assert clash_t[1] == pytest.approx(sc_clash_reward({"E_clash_total": 150.0}))
        assert aar_t == pytest.approx([0.7, 0.3])
        # each metric group contributes its own keys
        assert "shape/sc_mean" in metrics
        assert "sc_clash/e_total_mean" in metrics
        assert "aar/aar_mean" in metrics


class TestComputeRewardsRepackGate:
    def _cfg(self, **overrides):
        cfg = dict(
            w_iptm=0.0,
            w_ptm=0.0,
            w_abag_iptm=0.0,
            w_plddt=0.0,
            w_gpde=0.0,
            w_pae_global=0.0,
            w_pae_interface=0.0,
            w_sctm_binder=0.0,
            w_sctm_complex=0.0,
            log_struct_diagnostic=False,
            log_dist_diagnostic=False,
            per_token_clash=False,
            per_token_chainbreak=False,
            w_seq_diversity=0.0,
            w_struct_diversity=0.0,
            w_aa_dist=0.0,
            w_3di_dist=0.0,
            w_clash_contact=0.0,
            w_chainbreak=0.0,
            w_shape=0.0,
            w_sc_clash=0.0,
            w_aar=0.0,
        )
        cfg.update(overrides)
        return SimpleNamespace(**cfg)

    def _mk(self, **cfg):
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = self._cfg(**cfg)
        trainer.reward_client = Mock()
        trainer.reward_client.score_group.side_effect = AssertionError("score_group must not be called")
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 4, 3, 3))
        trainer._shape_terms_for_group = Mock(side_effect=AssertionError("_shape_terms_for_group must not be called"))
        trainer._repack_terms_for_group = Mock(side_effect=AssertionError("_repack_terms_for_group must not be called"))
        return trainer

    def test_all_zero_is_inert(self):
        """No repack weight ⇒ neither helper called, no decode, no repack metrics, reward 0."""
        trainer = self._mk()
        trainer._decode_backbone_coords = Mock(side_effect=AssertionError("must not decode"))
        rewards, metrics, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["AA", "CC", "DD"], tri_seqs=None, trajectory={}, comp={}
        )
        trainer._shape_terms_for_group.assert_not_called()
        trainer._repack_terms_for_group.assert_not_called()
        assert rewards.tolist() == pytest.approx([0.0, 0.0, 0.0])
        assert not any(k.startswith(("shape/", "sc_clash/", "aar/")) for k in metrics)

    def test_sc_only_routes_through_shape_helper(self):
        """w_shape only ⇒ byte-identical SC path (_shape_terms_for_group), NOT _repack."""
        trainer = self._mk(w_shape=1.0)
        trainer._shape_terms_for_group = Mock(
            return_value=([0.2, 0.6, 0.9], {"reward/shape_term_mean": 0.5666, "shape/sc_mean": 0.4})
        )
        rewards, metrics, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["AA", "CC", "DD"], tri_seqs=None, trajectory={}, comp={}
        )
        trainer._shape_terms_for_group.assert_called_once()
        trainer._repack_terms_for_group.assert_not_called()
        assert rewards.tolist() == pytest.approx([0.2, 0.6, 0.9])
        assert metrics["shape/sc_mean"] == pytest.approx(0.4)

    def test_sc_clash_only_routes_through_repack_with_clash_want(self):
        trainer = self._mk(w_sc_clash=1.0)
        trainer._repack_terms_for_group = Mock(
            return_value=(
                [0.0, 0.0, 0.0],
                [0.3, 0.5, 0.7],
                [0.0, 0.0, 0.0],
                {"reward/sc_clash_term_mean": 0.5},
                None,
                None,
            )
        )
        rewards, metrics, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["AA", "CC", "DD"], tri_seqs=None, trajectory={}, comp={}
        )
        trainer._shape_terms_for_group.assert_not_called()
        trainer._repack_terms_for_group.assert_called_once()
        args = trainer._repack_terms_for_group.call_args[0]
        assert args[4] == ("clash",)  # want: (target_id, trajectory, comp, seqs, want, ...)
        assert rewards.tolist() == pytest.approx([0.3, 0.5, 0.7])

    def test_shape_plus_clash_sums_both_terms(self):
        trainer = self._mk(w_shape=1.0, w_sc_clash=1.0)
        trainer._repack_terms_for_group = Mock(
            return_value=([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.0, 0.0, 0.0], {}, None, None)
        )
        rewards, *_ = trainer._compute_rewards(
            target_id="t0", seqs=["AA", "CC", "DD"], tri_seqs=None, trajectory={}, comp={}
        )
        trainer._shape_terms_for_group.assert_not_called()
        args = trainer._repack_terms_for_group.call_args[0]
        assert args[4] == ("sc", "clash")  # want built in (sc, clash, aar) order
        assert rewards.tolist() == pytest.approx([0.5, 0.7, 0.9])  # shape + clash summed
