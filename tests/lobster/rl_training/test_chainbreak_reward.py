"""Tests for the backbone chain-break (peptide-bond integrity) reward.

Two layers:

* pure-numpy unit tests of :func:`chainbreak_reward` — the intact→1.0 baseline, the
  ``mean_r·gate`` decomposition, the ``cap`` saturation lever (a 70 Å break scores the
  same as a 3.5 Å one), the ``return_eres`` per-residue split that sums to ``pen`` and
  recovers ``mean_r``, dense-array adjacency (a gap severs no bond), and the two gate
  modes (``count`` vs ``soft``);
* trainer-wiring tests of ``_chainbreak_terms_for_group`` / ``_struct_pos_advantage`` —
  the term is weighted, aggregated, scatters per-residue credit to ``(G, L)``, and the
  per-token chain-break advantage combines additively with the clash advantage while
  leaving the clash-only path byte-identical.

The trainer tests live here (not in ``test_trainers.py``) so they avoid importing
``lobster.rl_training.trainers`` (which pulls in ``trl``, absent in this env).
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import lobster.rl_training.rewards as _rewards
from lobster.rl_training import LeFlurGRPOTrainer
from lobster.rl_training.rewards import chainbreak_reward
from lobster.rl_training.rewards._chainbreak_reward import _r_bond, _sigmoid


# ------------------------------------------------------------------ fixtures
def _bb_chain(n: int, cn: float = 1.33) -> np.ndarray:
    """A straight ``n``-residue backbone with every ``C(i)–N(i+1)`` distance == ``cn``.

    Residues are laid along x with ``N``/``CA``/``C`` at fixed intra-residue offsets;
    the next residue's ``N`` is placed exactly ``cn`` Å past this residue's ``C``, so
    all peptide bonds share the length ``cn`` (``cn=1.33`` ⇒ ideal ⇒ ``r_bond=1``).
    """
    coords = np.zeros((n, 3, 3))
    x = 0.0
    for i in range(n):
        coords[i, 0] = [x, 0.0, 0.0]  # N
        coords[i, 1] = [x + 1.46, 0.0, 0.0]  # CA
        coords[i, 2] = [x + 2.44, 0.0, 0.0]  # C
        x = x + 2.44 + cn  # next N sits cn past this C
    return coords


def _break_bond(coords: np.ndarray, k: int, d: float) -> np.ndarray:
    """Return a copy with bond ``k`` (``C(k)→N(k+1)``) stretched to length ``d``.

    Only residue ``k+1``'s ``N`` atom is moved, so exactly one peptide bond changes
    (bond ``k+1`` uses residue ``k+1``'s ``C``, which is untouched).
    """
    out = coords.copy()
    out[k + 1, 0] = out[k, 2] + np.array([d, 0.0, 0.0])
    return out


def _all_binder(coords: np.ndarray):
    """(coords, valid_mask, binder_mask) with every residue a valid binder residue."""
    n = coords.shape[0]
    return coords, np.ones(n, dtype=bool), np.ones(n, dtype=bool)


# --------------------------------------------------------- unit: helpers
class TestHelpers:
    def test_sigmoid_matches_logistic_and_is_overflow_safe(self):
        x = np.array([-50.0, -1.0, 0.0, 1.0, 700.0])  # 700 overflows a naive exp(x)
        # matches the logistic on moderate inputs
        mod = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(_sigmoid(mod), 1.0 / (1.0 + np.exp(-mod)), rtol=1e-12)
        out = _sigmoid(x)
        assert np.all(np.isfinite(out))  # no overflow / nan on the severed-bond extreme
        assert out[0] == pytest.approx(0.0, abs=1e-12)
        assert out[-1] == pytest.approx(1.0, abs=1e-12)

    def test_r_bond_ideal_is_one_inside_deadband(self):
        d = np.array([1.33, 1.33 + 0.05, 1.33 - 0.05])  # within tol=0.10
        np.testing.assert_allclose(_r_bond(d, ideal=1.33, tol=0.10, cap=2.0, sigma=0.5), 1.0)

    def test_r_bond_saturates_at_cap(self):
        """Beyond |d-ideal|-tol == cap, every distance shares the same floor r_bond."""
        floor = np.exp(-((2.0 / 0.5) ** 2))
        for d in (3.5, 10.0, 70.0):  # all have excess == cap == 2.0
            assert _r_bond(np.array([d]), ideal=1.33, tol=0.10, cap=2.0, sigma=0.5)[0] == pytest.approx(floor)


# ---------------------------------------------------- unit: chainbreak reward
class TestChainbreakReward:
    def test_intact_backbone_scores_one(self):
        """An ideal-bond chain ⇒ mean_r 1, count gate 1, term 1, no hard breaks."""
        term, diag = chainbreak_reward(*_all_binder(_bb_chain(6)))
        assert term == pytest.approx(1.0)
        assert diag["mean_r"] == pytest.approx(1.0)
        assert diag["gate"] == pytest.approx(1.0)
        assert diag["n_hardbreak"] == 0
        assert diag["n_bonds"] == 5
        assert diag["pen"] == pytest.approx(0.0)
        assert diag["max_cn"] == pytest.approx(1.33)

    def test_single_break_lowers_term_and_gate(self):
        """One severed bond ⇒ term < 1, gate < 1, exactly one hard break flagged."""
        coords = _break_bond(_bb_chain(6), k=2, d=8.0)
        term, diag = chainbreak_reward(*_all_binder(coords))
        assert diag["n_hardbreak"] == 1
        assert diag["max_cn"] == pytest.approx(8.0)
        assert diag["gate"] == pytest.approx(np.exp(-1.0 / 2.0))  # count gate, gate_k=2
        assert diag["mean_r"] < 1.0
        assert term == pytest.approx(diag["mean_r"] * diag["gate"])
        assert 0.0 <= term < 1.0

    def test_more_breaks_score_lower(self):
        """Monotone: more severed bonds ⇒ strictly lower term (gate compounds)."""
        one = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(6), k=2, d=8.0)))[0]
        two = _break_bond(_bb_chain(6), k=1, d=8.0)
        two = _break_bond(two, k=3, d=8.0)
        two_term = chainbreak_reward(*_all_binder(two))[0]
        assert two_term < one < 1.0

    def test_cap_bounds_catastrophic_break(self):
        """The stability lever: a 70 Å break scores the same term as a 3.5 Å one."""
        near = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(6), k=2, d=3.5)))[0]
        far = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(6), k=2, d=70.0)))[0]
        assert near == pytest.approx(far)

    def test_fewer_than_two_residues_is_unpenalized(self):
        for n in (0, 1):
            coords = np.zeros((n, 3, 3))
            term, diag = chainbreak_reward(coords, np.ones(n, bool), np.ones(n, bool))
            assert term == pytest.approx(1.0)
            assert diag["n_bonds"] == 0
            assert diag["pen"] == pytest.approx(0.0)

    def test_bounded_unit_interval_over_break_sweep(self):
        for d in np.linspace(1.33, 90.0, 40):
            term, _ = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(5), k=1, d=float(d))))
            assert -1e-9 <= term <= 1.0 + 1e-9


class TestAdjacency:
    """Peptide bonds are scored only between residues adjacent in the dense array."""

    def test_gap_severs_no_bond(self):
        """A masked/non-binder gap between two binder stretches is NOT a peptide bond.

        Even though the two stretches are far apart in space, the reward must not count
        a bond across the gap (dense-array analog of a chain / resSeq break)."""
        coords = _bb_chain(5)
        coords[3:] += np.array([100.0, 0.0, 0.0])  # translate the second stretch far away
        valid = np.array([True, True, False, True, True])
        binder = np.array([True, True, False, True, True])
        term, diag = chainbreak_reward(coords, valid, binder)
        # bonds: (0,1) and (3,4) only — the (1,3) pair is not adjacent in bpos
        assert diag["n_bonds"] == 2
        assert diag["n_hardbreak"] == 0
        assert term == pytest.approx(1.0)

    def test_only_binder_residues_scored(self):
        """Antigen residues never contribute a bond even if severed."""
        coords = _bb_chain(6)
        coords[0:2] = _break_bond(coords[0:3], k=0, d=50.0)[0:2]  # wreck an antigen bond
        valid = np.ones(6, bool)
        binder = np.array([False, False, True, True, True, True])
        _, diag = chainbreak_reward(coords, valid, binder)
        assert diag["n_bonds"] == 3  # only the 4 binder residues => 3 bonds
        assert diag["n_hardbreak"] == 0


class TestPerResidueCredit:
    """``return_eres`` yields a per-binder-residue penalty that recovers pen and mean_r."""

    def test_off_by_default_and_byte_identical(self):
        coords = _break_bond(_bb_chain(6), k=2, d=8.0)
        term_a, diag_a = chainbreak_reward(*_all_binder(coords))
        term_b, diag_b = chainbreak_reward(*_all_binder(coords), return_eres=True)
        assert "cb_break_res" not in diag_a
        assert term_a == term_b
        assert diag_a["pen"] == pytest.approx(diag_b["pen"])

    def test_eres_sums_to_pen_and_recovers_mean_r(self):
        coords = _break_bond(_bb_chain(6), k=2, d=8.0)
        _, diag = chainbreak_reward(*_all_binder(coords), return_eres=True)
        e_res = diag["cb_break_res"]
        assert e_res.shape == (6,)  # one entry per valid binder residue
        assert e_res.sum() == pytest.approx(diag["pen"], rel=1e-9, abs=1e-9)
        assert diag["mean_r"] == pytest.approx(1.0 - diag["pen"] / diag["n_bonds"])

    def test_break_credit_concentrates_on_endpoints(self):
        """The severed bond's penalty lands 50/50 on its two flanking residues."""
        coords = _break_bond(_bb_chain(6), k=2, d=8.0)
        _, diag = chainbreak_reward(*_all_binder(coords), return_eres=True)
        e_res = diag["cb_break_res"]
        # bond k=2 joins residues 2 and 3; those carry (nearly) all the penalty.
        assert e_res[2] == pytest.approx(e_res[3])
        assert e_res[2] + e_res[3] == pytest.approx(diag["pen"], rel=1e-6)
        assert e_res[0] == pytest.approx(0.0, abs=1e-9)

    def test_eres_scattered_to_valid_binder_order(self):
        """With a gap, cb_break_res is indexed by the valid-binder order (not full L)."""
        coords = _bb_chain(5)
        coords[3:] += np.array([100.0, 0.0, 0.0])
        valid = np.array([True, True, False, True, True])
        binder = np.array([True, True, False, True, True])
        _, diag = chainbreak_reward(coords, valid, binder, return_eres=True)
        assert diag["cb_break_res"].shape == (4,)  # 4 valid binder residues
        assert diag["cb_break_res"].sum() == pytest.approx(diag["pen"], abs=1e-12)


class TestGateModes:
    def test_count_intact_is_exactly_one_soft_has_small_floor(self):
        """count gate = 1.0 on a clean chain; soft applies a tiny sub-threshold floor."""
        args = _all_binder(_bb_chain(6))
        t_count = chainbreak_reward(*args, gate_mode="count")[0]
        t_soft = chainbreak_reward(*args, gate_mode="soft")[0]
        assert t_count == pytest.approx(1.0)
        assert 0.99 < t_soft < 1.0  # bulk floor haircut, but negligible

    def test_soft_break_count_is_severity_aware(self):
        """soft n_break rises smoothly through the threshold; count steps at 2.0 Å."""
        below = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(4), 1, 1.9)), gate_mode="soft")[1]
        above = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(4), 1, 2.1)), gate_mode="soft")[1]
        assert below["n_break"] < 0.5 < above["n_break"]  # smoothly crosses ~0.5 at d0=2.0
        # count mode: a hard step at the threshold
        c_below = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(4), 1, 1.9)), gate_mode="count")[1]
        c_above = chainbreak_reward(*_all_binder(_break_bond(_bb_chain(4), 1, 2.1)), gate_mode="count")[1]
        assert c_below["n_hardbreak"] == 0
        assert c_above["n_hardbreak"] == 1

    def test_soft_gate_has_no_cliff_at_threshold(self):
        """Sweeping a bond across 2.0 Å, the soft term is continuous (count is not)."""
        ds = np.linspace(1.7, 2.3, 61)
        soft = np.array(
            [chainbreak_reward(*_all_binder(_break_bond(_bb_chain(4), 1, float(d))), gate_mode="soft")[0] for d in ds]
        )
        # max successive step is small -> no jump discontinuity across the threshold
        assert np.abs(np.diff(soft)).max() < 0.02

    def test_invalid_gate_mode_raises(self):
        with pytest.raises(ValueError, match="gate_mode"):
            chainbreak_reward(*_all_binder(_break_bond(_bb_chain(4), 1, 5.0)), gate_mode="bogus")


# ---------------------------------------------------------- trainer wiring
def _cb_cfg(**overrides):
    cfg = dict(
        w_chainbreak=1.0,
        chainbreak_gate="count",
        chainbreak_gate_k=2.0,
        chainbreak_ideal=1.33,
        chainbreak_tol=0.10,
        chainbreak_cap=2.00,
        chainbreak_sigma=0.50,
        chainbreak_break_hard=2.0,
        chainbreak_break_d0=2.0,
        chainbreak_break_soft=0.10,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestChainbreakTermsForGroup:
    def test_weights_and_aggregates(self, monkeypatch):
        """Term is weighted by w_chainbreak and chainbreak/* diagnostics are averaged."""
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _cb_cfg(w_chainbreak=2.0)
        trainer._decode_backbone_coords = Mock(return_value=torch.zeros(3, 5, 3, 3))
        comp = {"mask": torch.ones(1, 5), "binder_positions": torch.ones(5)}

        diags = iter(
            [
                (1.0, {"mean_r": 1.0, "gate": 1.0, "n_break": 0.0, "n_hardbreak": 0, "max_cn": 1.33}),
                (0.5, {"mean_r": 0.8, "gate": 0.6, "n_break": 1.0, "n_hardbreak": 1, "max_cn": 8.0}),
                (0.2, {"mean_r": 0.6, "gate": 0.3, "n_break": 2.0, "n_hardbreak": 2, "max_cn": 40.0}),
            ]
        )
        monkeypatch.setattr(_rewards, "chainbreak_reward", lambda *a, **k: next(diags))

        weighted, metrics = trainer._chainbreak_terms_for_group(trajectory={}, comp=comp)

        assert weighted == pytest.approx([2.0, 1.0, 0.4])  # 2.0 * term
        assert metrics["reward/chainbreak_term_mean"] == pytest.approx((2.0 + 1.0 + 0.4) / 3)
        assert metrics["chainbreak/mean_r_mean"] == pytest.approx((1.0 + 0.8 + 0.6) / 3)
        assert metrics["chainbreak/gate_mean"] == pytest.approx((1.0 + 0.6 + 0.3) / 3)
        assert metrics["chainbreak/n_break_mean"] == pytest.approx((0.0 + 1.0 + 2.0) / 3)
        assert metrics["chainbreak/n_hardbreak_mean"] == pytest.approx((0 + 1 + 2) / 3)
        assert metrics["chainbreak/max_cn_mean"] == pytest.approx((1.33 + 8.0 + 40.0) / 3)
        # 2 of the 3 designs carry >=1 hard break
        assert metrics["chainbreak/frac_designs_broken"] == pytest.approx(2 / 3)

    def test_uses_shared_gen_bb_without_decoding(self):
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _cb_cfg()
        trainer._decode_backbone_coords = Mock(side_effect=AssertionError("must not decode"))
        comp = {"mask": torch.ones(1, 6), "binder_positions": torch.ones(6)}
        gen_bb = np.stack([_bb_chain(6), _break_bond(_bb_chain(6), 2, 8.0)])  # (2, 6, 3, 3)
        weighted, metrics = trainer._chainbreak_terms_for_group(trajectory={}, comp=comp, gen_bb=gen_bb)
        assert len(weighted) == 2
        assert weighted[0] == pytest.approx(1.0)  # intact
        assert weighted[1] < 1.0  # broken
        trainer._decode_backbone_coords.assert_not_called()


class TestChainbreakTermsPerResidue:
    """``_chainbreak_terms_for_group(return_eres=True)`` scatters per-residue penalty to (G, L)."""

    def test_return_eres_scatters_and_sums(self, monkeypatch):
        trainer = object.__new__(LeFlurGRPOTrainer)
        trainer.device = "cpu"
        trainer.config = _cb_cfg(w_chainbreak=1.0)
        # L=5: positions 0-1 antigen, 2-4 binder (all valid).
        comp = {"mask": torch.ones(1, 5), "binder_positions": torch.tensor([0, 0, 1, 1, 1])}
        gen_bb = np.zeros((2, 5, 3, 3), dtype=np.float32)
        diags = iter(
            [
                (
                    0.7,
                    {
                        "mean_r": 0.9,
                        "gate": 0.78,
                        "n_break": 0.0,
                        "n_hardbreak": 0,
                        "max_cn": 1.4,
                        "pen": 0.5,
                        "cb_break_res": np.array([0.1, 0.2, 0.2]),
                    },
                ),
                (
                    0.3,
                    {
                        "mean_r": 0.7,
                        "gate": 0.43,
                        "n_break": 1.0,
                        "n_hardbreak": 1,
                        "max_cn": 9.0,
                        "pen": 1.0,
                        "cb_break_res": np.array([0.5, 0.5, 0.0]),
                    },
                ),
            ]
        )
        monkeypatch.setattr(_rewards, "chainbreak_reward", lambda *a, **k: next(diags))

        weighted, metrics, e_res_full = trainer._chainbreak_terms_for_group(
            trajectory={}, comp=comp, gen_bb=gen_bb, return_eres=True
        )
        assert e_res_full.shape == (2, 5)
        assert np.all(e_res_full[:, :2] == 0.0)  # antigen positions carry no break penalty
        np.testing.assert_allclose(e_res_full[0, 2:], [0.1, 0.2, 0.2])
        np.testing.assert_allclose(e_res_full[1, 2:], [0.5, 0.5, 0.0])
        # per-design row-sum equals pen for that design
        np.testing.assert_allclose(e_res_full.sum(axis=1), [0.5, 1.0])
        assert weighted == pytest.approx([0.7, 0.3])  # w_chainbreak=1.0


class TestStructPosAdvantageChainbreak:
    """Per-position structure advantage with the chain-break per-residue signal.

    Mirrors the clash cases in ``test_clash_reward.py`` but drives the chain-break
    keyword arm, then checks the two signals compose additively and that the clash-only
    call is left untouched (chainbreak_eres defaults None)."""

    def _trainer(self, *, w_pt_clash: float = 1.0, w_pt_chainbreak: float = 1.0):
        t = object.__new__(LeFlurGRPOTrainer)
        t.device = "cpu"
        t.config = SimpleNamespace(w_pt_clash=w_pt_clash, w_pt_chainbreak=w_pt_chainbreak, adv_eps=1e-6)
        return t

    def test_zero_break_reduces_to_design_adv(self):
        t = self._trainer()
        A = t._struct_pos_advantage(
            None, torch.tensor([1.0, -0.5, 0.25]), torch.ones(4), chainbreak_eres=torch.zeros(3, 4)
        )
        assert A.shape == (3, 4)
        torch.testing.assert_close(A, torch.tensor([1.0, -0.5, 0.25]).unsqueeze(1).expand(3, 4).contiguous())

    def test_lower_break_gets_higher_advantage(self):
        t = self._trainer()
        cb = torch.tensor([[0.0, 0.0], [10.0, 0.0]])  # design 0 breaks less at position 0
        A = t._struct_pos_advantage(None, torch.zeros(2), torch.ones(2), chainbreak_eres=cb)
        assert A[0, 0] > A[1, 0]  # less break -> higher advantage
        assert A[0, 1] == pytest.approx(A[1, 1])  # equal at position 1

    def test_offmask_position_has_no_break_credit(self):
        t = self._trainer()
        cb = torch.tensor([[5.0, 1.0], [0.0, 2.0]])
        design_adv = torch.tensor([0.3, -0.2])
        A = t._struct_pos_advantage(None, design_adv, torch.tensor([1.0, 0.0]), chainbreak_eres=cb)
        torch.testing.assert_close(A[:, 1], design_adv)  # only broadcast design adv survives

    def test_w_pt_chainbreak_scales_credit_linearly(self):
        cb = torch.tensor([[0.0, 0.0], [10.0, 4.0]])
        A1 = self._trainer(w_pt_chainbreak=1.0)._struct_pos_advantage(
            None, torch.zeros(2), torch.ones(2), chainbreak_eres=cb
        )
        A2 = self._trainer(w_pt_chainbreak=2.0)._struct_pos_advantage(
            None, torch.zeros(2), torch.ones(2), chainbreak_eres=cb
        )
        torch.testing.assert_close(A2, 2.0 * A1)

    def test_clash_and_chainbreak_are_additive(self):
        """Both arms on == clash-only advantage + chainbreak-only advantage (design_adv counted once)."""
        t = self._trainer(w_pt_clash=1.0, w_pt_chainbreak=1.0)
        clash = torch.tensor([[0.0, 3.0], [6.0, 1.0]])
        cb = torch.tensor([[2.0, 0.0], [0.0, 5.0]])
        design_adv = torch.tensor([0.4, -0.1])
        mask = torch.ones(2)
        both = t._struct_pos_advantage(clash, design_adv, mask, chainbreak_eres=cb)
        clash_only = t._struct_pos_advantage(clash, design_adv, mask)
        cb_only = t._struct_pos_advantage(None, design_adv, mask, chainbreak_eres=cb)
        design_bcast = design_adv.unsqueeze(1).expand(2, 2)
        torch.testing.assert_close(both, clash_only + cb_only - design_bcast)

    def test_clash_only_call_is_unchanged_by_refactor(self):
        """The positional clash-only signature (chainbreak_eres default None) still works."""
        t = self._trainer(w_pt_clash=1.0)
        clash = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        A = t._struct_pos_advantage(clash, torch.zeros(2), torch.ones(2))
        assert A[0, 0] > A[1, 0]
        assert A[0, 1] == pytest.approx(A[1, 1])
