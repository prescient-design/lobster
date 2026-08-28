"""Behavior tests for the tmol interface-ΔΔG reward (:mod:`plm_design_rl.rewards._ddg_reward`).

The scalar reward mapper is pure math and always runs. The scoring engine needs the optional
``tmol`` extra; when it is absent we assert the *import discipline* instead — the module still
imports, and the first scoring call raises a clear ``pip install plm-design-rl[tmol]`` error
rather than a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib.util

import pytest

from plm_design_rl.rewards import _ddg_reward as ddg

_HAS_TMOL = importlib.util.find_spec("tmol") is not None


def test_reward_bounds_and_monotonicity() -> None:
    # more-negative ΔΔG (stronger binding) => higher reward, bounded to [0, 1]
    strong = ddg.ddg_reward({"ddg_total": -80.0})
    mid = ddg.ddg_reward({"ddg_total": ddg.DDG_CENTER})
    weak = ddg.ddg_reward({"ddg_total": +80.0})
    assert 0.0 <= weak < mid < strong <= 1.0
    assert mid == pytest.approx(0.5, abs=1e-6)  # reward == 0.5 at the center


def test_reward_missing_and_nonfinite_floor_to_zero() -> None:
    assert ddg.ddg_reward(None) == 0.0
    assert ddg.ddg_reward({}) == 0.0  # no ddg_total key
    assert ddg.ddg_reward({"ddg_total": float("nan")}) == 0.0
    assert ddg.ddg_reward({"ddg_total": float("inf")}) == 0.0


def test_reward_flavor_and_key_selection() -> None:
    res = {"ddg_total": +80.0, "ddg_ub_total": -80.0, "ddg_noclash": -80.0}
    # unbound-relaxed flavor reads the ddg_ub_* column
    assert ddg.ddg_reward(res, flavor="ddg") < 0.5 < ddg.ddg_reward(res, flavor="ddg_ub")
    # alternate group suffix
    assert ddg.ddg_reward(res, key="noclash") > 0.5


def test_reward_never_overflows_at_extremes() -> None:
    # extreme values must not raise (numerically-stable logistic) and stay in-range
    assert ddg.ddg_reward({"ddg_total": -1e9}) == pytest.approx(1.0)
    assert ddg.ddg_reward({"ddg_total": 1e9}) == pytest.approx(0.0)


@pytest.mark.skipif(_HAS_TMOL, reason="tmol installed; import-discipline error not raised")
def test_ddg_terms_raises_install_hint_without_tmol() -> None:
    with pytest.raises(ImportError, match=r"plm-design-rl\[tmol\]"):
        ddg.ddg_terms("/nonexistent.pdb")
    with pytest.raises(ImportError, match=r"plm-design-rl\[tmol\]"):
        ddg.ddg_per_binder_residue("/nonexistent.pdb")


@pytest.mark.skipif(_HAS_TMOL, reason="tmol installed; import-discipline error not raised")
def test_ddg_packed_all_raises_install_hint_without_tmol() -> None:
    # The single-pass engine the GRPO repack worker calls ("ddg" want) must obey the same lazy-
    # import discipline: no tmol at import time, a clear install-the-extra error only on first use.
    with pytest.raises(ImportError, match=r"plm-design-rl\[tmol\]"):
        ddg.ddg_packed_all("/nonexistent.pdb")


def test_ddg_packed_all_is_exported() -> None:
    # Reachable from the package and byte-identical via the lobster back-compat shim, so the
    # worker's ``from plm_design_rl.rewards._ddg_reward import ddg_packed_all`` and any
    # ``from lobster.rl_training.rewards import ddg_packed_all`` resolve to the same callable.
    from lobster.rl_training.rewards import ddg_packed_all as shim_fn
    from plm_design_rl.rewards import ddg_packed_all as pkg_fn

    assert pkg_fn is ddg.ddg_packed_all
    assert shim_fn is ddg.ddg_packed_all
    assert callable(ddg.ddg_packed_all)
