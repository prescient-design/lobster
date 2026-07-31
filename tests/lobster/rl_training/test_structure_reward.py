"""Tests for the structure self-consistency TM-score reward (``_structure_reward.py``).

The term measures how well the sequence the policy designed folds back to the
backbone it generated: a Kabsch superposition + vendored TM-score between the
LeFlur-decoded CA set and the Protenix-predicted CA set. All pure numpy, so no GPU
/ Protenix needed. See ``rewards/README.md`` §2.
"""

from __future__ import annotations

import numpy as np
import pytest

from lobster.rl_training.rewards._structure_reward import (
    kabsch,
    structure_terms,
    tm_score,
)


def _rand_coords(n: int, seed: int = 0) -> np.ndarray:
    # Spread points out so d0-scale distances are meaningful for a real backbone.
    return np.random.RandomState(seed).randn(n, 3) * 10.0


def _rotation(seed: int = 1) -> np.ndarray:
    # A proper rotation matrix (det +1) via QR of a random matrix.
    q, r = np.linalg.qr(np.random.RandomState(seed).randn(3, 3))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_tm_identity_is_one() -> None:
    P = _rand_coords(60)
    assert tm_score(P, P) == pytest.approx(1.0)


def test_tm_kabsch_rotation_translation_invariant() -> None:
    # A rigidly moved copy superposes exactly -> TM ~ 1 (Kabsch removes the transform).
    P = _rand_coords(80, seed=2)
    R = _rotation(seed=3)
    Q = P @ R.T + np.array([12.0, -5.0, 7.0])
    assert tm_score(P, Q) == pytest.approx(1.0, abs=1e-6)


def test_tm_random_is_low() -> None:
    # Two unrelated point clouds superpose poorly -> low TM (well below 0.5).
    a = _rand_coords(100, seed=4)
    b = _rand_coords(100, seed=5)
    assert tm_score(a, b) < 0.5


def test_tm_length_mismatch_and_empty() -> None:
    assert tm_score(_rand_coords(10), _rand_coords(9)) == 0.0
    assert tm_score(np.zeros((0, 3)), np.zeros((0, 3))) == 0.0


def test_kabsch_moves_p_onto_q() -> None:
    P = _rand_coords(40, seed=6)
    R = _rotation(seed=7)
    Q = P @ R.T + np.array([1.0, 2.0, 3.0])
    P_sup = kabsch(P, Q)
    assert np.allclose(P_sup, Q, atol=1e-6)


def test_kabsch_shape_validation() -> None:
    with pytest.raises(ValueError):
        kabsch(_rand_coords(10), _rand_coords(9))


def test_structure_terms_identity() -> None:
    binder = _rand_coords(50, seed=8)
    complex_ = _rand_coords(90, seed=9)
    st = structure_terms(binder, binder, complex_, complex_)
    assert st["sctm_binder"] == pytest.approx(1.0)
    assert st["sctm_complex"] == pytest.approx(1.0)


def test_structure_terms_rigid_invariant() -> None:
    binder = _rand_coords(50, seed=10)
    R = _rotation(seed=11)
    pred_binder = binder @ R.T + np.array([3.0, 3.0, 3.0])
    st = structure_terms(binder, pred_binder, None, None)
    assert st["sctm_binder"] == pytest.approx(1.0, abs=1e-6)
    assert st["sctm_complex"] == 0.0  # complex pair missing -> 0


def test_structure_terms_missing_and_mismatch_are_zero() -> None:
    binder = _rand_coords(50, seed=12)
    # Missing pred -> 0; length-mismatched pair -> 0.
    assert structure_terms(binder, None, None, None) == {"sctm_binder": 0.0, "sctm_complex": 0.0}
    st = structure_terms(binder, _rand_coords(49, seed=13), None, None)
    assert st["sctm_binder"] == 0.0
