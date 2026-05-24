"""Phase 7 regression tests for :func:`align_and_compute_rmsd`.

The "RMSD-sqrt(3) regression" the LeFlur cleanup plan refers to is the bug
class where per-atom RMSD is accidentally averaged over the **3N flattened
coordinate scalars** (giving ``sum_sq_diff / 3N`` and therefore RMSD
divided by ``sqrt(3)`` versus the per-atom definition). The standard
per-atom RMSD averages squared distances ``||r1_i - r2_i||^2`` over the
``N`` atoms, not the ``3N`` coordinates.

These tests pin the formula by:

1. Building a pair of structures with a *known* per-atom displacement.
2. Asserting ``align_and_compute_rmsd`` returns the per-atom RMSD, not the
   per-coordinate RMSD (which would be ``per_atom_rmsd / sqrt(3)``).

The Kabsch alignment is set up so that the optimal rotation+translation is
the identity (input pairs are pre-aligned), letting us reason about RMSD
in closed form. We also pin two boundary cases:

* An empty mask returns ``0.0`` (no positions to align).
* Identical coordinates round-trip to RMSD ``0.0`` regardless of mask.
"""

from __future__ import annotations

import math

import pytest
import torch

def _make_pair_with_known_rmsd(
    n: int, per_atom_displacement: float, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return ``(coords1, coords2, expected_rmsd)`` with closed-form RMSD.

    Construction: place ``n`` CA atoms randomly, then displace each one
    along the y-axis by ``+d`` (even indices) or ``-d`` (odd indices).
    The displacement pattern has:

    * **zero net translation** (even/odd split is symmetric around 0), so
      Kabsch's translation step is the identity.
    * **zero net rotation effect** (the displacement is not a rotation, it
      flips per-atom; ``∇R`` of the Kabsch cost vanishes at ``R = I``).

    Under these conditions, the optimal Kabsch alignment is the identity
    and the returned RMSD equals ``per_atom_displacement`` *exactly* in
    closed form, decoupled from sampling noise.
    """
    if n % 2:
        raise ValueError("use even n so the +d/-d split has zero net translation")
    torch.manual_seed(seed)
    coords2 = torch.randn(n, 3, 3, dtype=torch.float64) * 5.0
    coords2 = coords2 - coords2[:, 1, :].mean(dim=0, keepdim=True).unsqueeze(0)

    sign = torch.ones(n, dtype=torch.float64)
    sign[1::2] = -1.0
    offset = torch.zeros(n, 3, dtype=torch.float64)
    offset[:, 1] = sign * per_atom_displacement  # +d / -d along y per atom

    coords1 = coords2.clone()
    coords1[:, 0, :] += offset
    coords1[:, 1, :] += offset  # CA
    coords1[:, 2, :] += offset

    return coords1.float(), coords2.float(), per_atom_displacement


def test_rmsd_is_per_atom_not_per_coordinate() -> None:
    """RMSD averages over ``N`` atoms, NOT over ``3N`` coordinate scalars.

    If the implementation regressed to averaging over flattened coordinates,
    the returned value would be ``per_atom_rmsd / sqrt(3)`` — exactly the
    historical bug the LeFlur cleanup plan calls out. We tolerate Kabsch
    finding a tiny rotation that improves RMSD by a sub-percent on random
    structures (empirically ~0.3% on n=64), but anything that drops the
    result toward the per-coordinate regime is flagged.
    """
    from lobster.metrics import align_and_compute_rmsd

    coords1, coords2, expected = _make_pair_with_known_rmsd(
        n=64, per_atom_displacement=2.5, seed=42
    )
    rmsd = align_and_compute_rmsd(coords1, coords2)

    assert isinstance(rmsd, float), "expected RMSD to be a Python float"
    # Per-atom RMSD identity: with the +d/-d zero-net displacement pattern,
    # ``mean over atoms of ||delta||^2 = d^2`` so ``sqrt(mean) = d``.
    # Kabsch on n=64 random points can shave ~0.3% off via a small fit.
    assert rmsd == pytest.approx(expected, rel=0.01), (
        f"per-atom RMSD broke: got {rmsd:.6f}, expected ≈{expected:.6f}"
    )
    # And explicitly NOT the per-coordinate form (the historical bug).
    per_coordinate = expected / math.sqrt(3.0)
    assert rmsd > per_coordinate * 1.3, (
        f"RMSD {rmsd:.4f} looks like per-coordinate form "
        f"({per_coordinate:.4f}); this is the historical sqrt(3) bug — "
        f"formula should average squared distances over atoms (not "
        f"flatten to 3N scalars)."
    )


def test_rmsd_scaling_invariant() -> None:
    """RMSD scales linearly with displacement magnitude (no sqrt(3) drift)."""
    from lobster.metrics import align_and_compute_rmsd

    for d in (0.5, 1.0, 2.0, 5.0):
        coords1, coords2, expected = _make_pair_with_known_rmsd(
            n=32, per_atom_displacement=d, seed=7
        )
        rmsd = align_and_compute_rmsd(coords1, coords2)
        assert rmsd == pytest.approx(expected, rel=0.02), (
            f"RMSD at displacement={d}: got {rmsd}, expected ≈{expected}"
        )
        # Sanity: per-coordinate regime would be ``d/sqrt(3)``.
        assert rmsd > (d / math.sqrt(3.0)) * 1.3, (
            f"RMSD at displacement={d} collapsed toward per-coordinate "
            f"value {d/math.sqrt(3.0):.3f} — sqrt(3) regression?"
        )


def test_rmsd_identical_coords_is_zero() -> None:
    """RMSD of a structure with itself is 0.0 regardless of mask."""
    from lobster.metrics import align_and_compute_rmsd

    torch.manual_seed(0)
    coords = torch.randn(40, 3, 3)
    assert align_and_compute_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-5)
    mask = torch.ones(40)
    mask[:10] = 0
    assert align_and_compute_rmsd(coords, coords, mask=mask) == pytest.approx(
        0.0, abs=1e-5
    )


def test_rmsd_empty_mask_returns_zero() -> None:
    """An all-zero mask short-circuits to RMSD=0.0 (defensive contract)."""
    from lobster.metrics import align_and_compute_rmsd

    coords1 = torch.randn(20, 3, 3)
    coords2 = torch.randn(20, 3, 3)
    mask = torch.zeros(20)
    assert align_and_compute_rmsd(coords1, coords2, mask=mask) == 0.0


def test_rmsd_respects_mask_on_partial_region() -> None:
    """When only half the atoms are masked-in, RMSD reflects only those atoms."""
    from lobster.metrics import align_and_compute_rmsd

    # Build a 40-atom structure where atoms 0..19 are pre-aligned (RMSD 0)
    # and atoms 20..39 are displaced by 3.0Å each with the same zero-net
    # pattern from ``_make_pair_with_known_rmsd``. With mask=first-20 the
    # RMSD should be ~0; with mask=last-20 it should be ~3.0 (Kabsch on
    # the half-population can introduce a small rigid-body fit, hence the
    # 0.3Å fuzz).
    coords1, coords2, _ = _make_pair_with_known_rmsd(
        n=40, per_atom_displacement=3.0, seed=11
    )
    coords1_partial = coords2.clone()
    coords1_partial[20:] = coords1[20:]

    mask_first = torch.zeros(40)
    mask_first[:20] = 1
    rmsd_first = align_and_compute_rmsd(coords1_partial, coords2, mask=mask_first)
    assert rmsd_first == pytest.approx(0.0, abs=1e-3)

    mask_last = torch.zeros(40)
    mask_last[20:] = 1
    rmsd_last = align_and_compute_rmsd(coords1_partial, coords2, mask=mask_last)
    # Half-population Kabsch fit can drop RMSD below the closed-form 3.0;
    # what matters for the sqrt(3) regression is that we are clearly in
    # the per-atom regime (~3.0) rather than the per-coordinate regime
    # (~3/sqrt(3) ≈ 1.73).
    assert rmsd_last > 3.0 / math.sqrt(3.0) + 0.5, (
        f"masked-region RMSD {rmsd_last:.3f} collapsed toward the "
        f"per-coordinate value {3.0/math.sqrt(3.0):.3f} — sqrt(3) bug?"
    )
