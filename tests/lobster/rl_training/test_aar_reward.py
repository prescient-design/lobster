"""Tests for the whole-binder LigandMPNN amino-acid-recovery (AAR / C_mpnn) reward.

Covers the pure-numpy layer of :mod:`lobster.rl_training.rewards._aar_reward`:

* ``reward_from_aar`` — the worker-result → float mapper (whole-binder AAR, floored to
  0.0 on ``None`` / missing / ``nan``, clamped to ``[0, 1]``),
* ``interface_residue_mask`` — the binder-residue interface selector (diagnostic only), and
* ``aar_terms`` — the aggregator that produces the whole-binder ``aar`` reward input plus the
  interface-restricted ``aar_iface`` / ``c_mpnn`` diagnostics.

The reward SIGNAL is the whole-binder recovery; the interface figures are tracked as
diagnostics only (never the optimisation target). AAR is anti-predictive of pass offline,
so the term ships opt-in (default weight 0) — these tests pin the math, not a pass gate.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lobster.rl_training.rewards import reward_from_aar
from lobster.rl_training.rewards._aar_reward import aar_terms, interface_residue_mask


class TestRewardFromAar:
    def test_none_missing_nan_floor_to_zero(self):
        assert reward_from_aar(None) == 0.0
        assert reward_from_aar({}) == 0.0
        assert reward_from_aar({"aar": float("nan")}) == 0.0

    def test_passthrough(self):
        assert reward_from_aar({"aar": 0.42}) == pytest.approx(0.42)

    def test_clamped_to_unit_interval(self):
        assert reward_from_aar({"aar": 1.5}) == pytest.approx(1.0)
        assert reward_from_aar({"aar": -0.5}) == pytest.approx(0.0)

    def test_monotone(self):
        assert reward_from_aar({"aar": 0.2}) < reward_from_aar({"aar": 0.8})


class TestInterfaceResidueMask:
    def test_selects_near_binder_residues(self):
        # binder residues at x=0,5,50; antigen residue at x=1 -> only binder res 0 within 8A
        bd_ca = np.array([[0.0, 0, 0], [5.0, 0, 0], [50.0, 0, 0]])
        ag_ca = np.array([[1.0, 0, 0]])
        mask = interface_residue_mask(bd_ca, ag_ca, d0=8.0)
        assert mask.dtype == bool and mask.shape == (3,)
        assert mask.tolist() == [True, True, False]

    def test_none_within_cutoff(self):
        bd_ca = np.array([[0.0, 0, 0]])
        ag_ca = np.array([[100.0, 0, 0]])
        assert interface_residue_mask(bd_ca, ag_ca, d0=8.0).tolist() == [False]

    def test_empty_antigen(self):
        bd_ca = np.array([[0.0, 0, 0], [5.0, 0, 0]])
        mask = interface_residue_mask(bd_ca, np.zeros((0, 3)), d0=8.0)
        assert mask.tolist() == [False, False]


class TestAarTerms:
    def _inputs(self, L, binder_idx, matches):
        """match_res (L,) bool, logp_res (L,), binder_mask (L,) bool."""
        match_res = np.zeros(L, dtype=bool)
        for i, m in zip(binder_idx, matches):
            match_res[i] = m
        logp_res = np.full(L, -1.0)
        binder_mask = np.zeros(L, dtype=bool)
        binder_mask[binder_idx] = True
        return match_res, logp_res, binder_mask

    def test_whole_binder_recovery_is_mean_over_binder(self):
        # binder residues 2,3,4; 2 of 3 recovered -> aar = 2/3
        match_res, logp_res, binder_mask = self._inputs(6, [2, 3, 4], [True, True, False])
        t = aar_terms(match_res, logp_res, binder_mask)
        assert t["aar"] == pytest.approx(2 / 3)
        assert t["term"] == pytest.approx(2 / 3)
        assert t["n_binder"] == 3

    def test_iface_is_subset_diagnostic(self):
        match_res, logp_res, binder_mask = self._inputs(6, [2, 3, 4], [True, True, False])
        iface_mask = np.zeros(6, dtype=bool)
        iface_mask[[3, 4]] = True  # interface = residues 3,4 -> 1 of 2 recovered
        t = aar_terms(match_res, logp_res, binder_mask, iface_mask=iface_mask)
        assert t["aar"] == pytest.approx(2 / 3)  # whole-binder unchanged
        assert t["aar_iface"] == pytest.approx(0.5)
        assert t["n_iface"] == 2

    def test_no_binder_is_nan(self):
        match_res = np.zeros(4, dtype=bool)
        logp_res = np.full(4, -1.0)
        binder_mask = np.zeros(4, dtype=bool)
        t = aar_terms(match_res, logp_res, binder_mask)
        assert math.isnan(t["aar"])
        assert t["n_binder"] == 0

    def test_no_iface_gives_nan_iface(self):
        match_res, logp_res, binder_mask = self._inputs(6, [2, 3, 4], [True, True, False])
        iface_mask = np.zeros(6, dtype=bool)
        t = aar_terms(match_res, logp_res, binder_mask, iface_mask=iface_mask)
        assert not math.isnan(t["aar"])
        assert math.isnan(t["aar_iface"])
        assert t["n_iface"] == 0
