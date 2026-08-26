"""3D-Zernike interface shape-complementarity (SC) shaping reward for LeFlur GRPO.

The interface-distribution reward (:mod:`._distribution_reward`) matches the
*composition* of the interface and the clash reward (:mod:`._clash_reward`) scores
steric quality, but neither asks whether the two partners' interface surfaces
*fit together* geometrically. This module scores exactly that: the **shape
complementarity** of the antigen and binder contact patches, via rotation-invariant
3D Zernike descriptors (3DZD).

.. warning::

   **This reward is a validated-negative prototype; keep its weight at 0.** On
   *full-atom* Complexa designs the numpy-3DZD raw-Pearson SC separates Protenix
   pass/fail well (AUROC 0.730, standardized logistic β = 0.841 after controlling for
   binder length + patch size — not a size proxy; pass rate 11.8% → 54.6% across SC
   quartiles). **But at GRPO reward time LeFlur decodes only the N/CA/C backbone**, and
   re-running the identical pipeline on backbone-only atoms (job 19765822, same 3500
   designs) collapses the signal to AUROC 0.533 with β = −0.273 (wrong sign) and flat/
   U-shaped quartiles. Shape complementarity is a sidechain-packing phenomenon; the
   coarse backbone surface cannot carry it. This module is kept as the prototype of the
   approach and as documentation of the negative result — it must not be enabled as a
   live reward with backbone-only coords (it would add real per-rollout SASA + Zernike
   compute for ~zero discriminative signal). It would only become useful if sidechains
   were cheaply available at reward time (they are not). See memory
   ``zernike-sc-discriminates-pass``.

The Kihara ``map2zernike`` binary pipeline showed no signal even on full-atom patches,
so only the pure-numpy engine is ported here.

Like the other shaping terms this module is pure numpy/scipy (no torch / Protenix /
trl), unit-testable in isolation, and inert (contributes exactly 0) when its weight
is 0. It is used as a **scalar** in GRPO (advantage-weighted log-prob) — not
back-propagated — so it need not be autograd-differentiable.

Design → reward
---------------
For one design with decoded backbone ``coords_full (L, 3, 3)`` in ``[N, CA, C]``
order plus ``valid_mask`` / ``binder_mask`` ``(L,)``:

**Crop-safe contact patch (ΔSASA).** Build a Shrake-Rupley solvent-accessible dot
surface for the antigen and the binder *independently* (backbone atoms only — no
sidechains are decoded), then keep a chain-A dot iff it is (a) accessible on the
isolated chain **and** (b) buried by the partner in the complex. That single
criterion excludes the buried core, the crop/domain-boundary faces (the antigens are
cropped domains), and away-facing surface — the descriptor sees only the true
exposed contact surface, never the artificial crop face.

**3D Zernike descriptor.** Each patch dot cloud is centered + scaled into the unit
ball and reduced to an order-20 rotation-invariant 3DZD (121 moments
``F_nl = sqrt(Σ_m |Ω_nlm|²)``). Rotation/translation/scale invariance is exact by
construction (asserted in tests).

**SC term** ``= clip(pearson(F_antigen, F_binder), 0, 1) ∈ [0, 1]`` — high when the
two interface patches share a region-based shape signature (complementary
knob/hole), i.e. good shape fit. NaN (degenerate/empty patch) ⇒ 0. The trainer
multiplies by ``w_shape_sc``.

Notes
-----
Backbone-only patches are ~10× sparser than the full-atom patches used to *derive*
the AUROC=0.730 result, so the reward path is validated separately at reward-time
resolution (``scripts/_zernike_compute.py --backbone``). Keep ``nsphere`` /
``order`` here matched to whatever that validation used.

The pure-math core (SASA dots, 3DZD) is duplicated from
``scripts/_zernike_sc.py`` (the offline analysis engine) rather than imported, to
keep the training library self-contained (no ``scripts`` / ``biotite`` dependency);
the two must stay numerically identical.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_jacobi, sph_harm_y

# Backbone atom van der Waals radii (Bondi, Å) for the decoded [N, CA, C] atoms.
_BB_RADII = np.array([1.55, 1.70, 1.70], dtype=np.float64)  # N, CA(C), C

# Bondi van der Waals radii (Å) by element; fallback 1.70 for unknown elements.
# Kept numerically identical to ``scripts/_zernike_sc.py::VDW_RADII`` so the all-atom
# reward path reproduces the offline SC validation exactly.
VDW_RADII: dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "SE": 1.90,
    "ZN": 1.39,
    "MG": 1.73,
    "NA": 2.27,
    "CA": 2.31,
    "FE": 2.05,
    "MN": 2.05,
    "K": 2.75,
}
PROBE = 1.4  # water probe radius (Å)
DEFAULT_ORDER = 20  # 3DZD order -> 121 invariants
DEFAULT_NSPHERE = 192  # Shrake-Rupley dots per atom (matches offline validation)


def vdw_radius(element: str) -> float:
    """Bondi van der Waals radius (Å) for a chemical ``element`` (fallback 1.70)."""
    return VDW_RADII.get(element.strip().upper(), 1.70)


def radii_from_elements(elements) -> np.ndarray:
    """Map a sequence of element symbols to a ``(N,)`` array of vdw radii."""
    return np.array([vdw_radius(e) for e in elements], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Shrake-Rupley dot surface + crop-safe ΔSASA contact masking
# --------------------------------------------------------------------------- #


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` ~uniform unit vectors on S^2 (Fibonacci lattice)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    golden = np.pi * (1.0 + 5.0**0.5)
    theta = golden * i
    return np.stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)


def _accessible_dots(
    coords: np.ndarray, radii: np.ndarray, probe: float = PROBE, nsphere: int = DEFAULT_NSPHERE
) -> np.ndarray:
    """Solvent-accessible surface dots of an *isolated* atom cloud.

    A dot on atom ``i`` (``center_i + (r_i + probe)·unit``) survives iff it lies
    outside every other atom's ``(r_j + probe)`` ball — the exposed surface of the
    cloud on its own, excluding the buried core by construction.
    """
    if coords.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    sphere = _fibonacci_sphere(nsphere)
    R = radii + probe
    tree = cKDTree(coords)
    maxR = float(R.max())
    kept: list[np.ndarray] = []
    for i in range(coords.shape[0]):
        pts = coords[i] + R[i] * sphere
        idx = [j for j in tree.query_ball_point(coords[i], R[i] + maxR) if j != i]
        if idx:
            cj = coords[idx]
            Rj = R[idx]
            d2 = ((pts[:, None, :] - cj[None, :, :]) ** 2).sum(-1)
            buried = (d2 < (Rj[None, :] ** 2)).any(1)
        else:
            buried = np.zeros(pts.shape[0], dtype=bool)
        if not buried.all():
            kept.append(pts[~buried])
    return np.concatenate(kept, axis=0) if kept else np.empty((0, 3), dtype=np.float64)


def _buried_by(dots: np.ndarray, coords_other: np.ndarray, radii_other: np.ndarray, probe: float = PROBE) -> np.ndarray:
    """Boolean mask: dot lies inside the ``(r + probe)`` ball union of the partner."""
    if dots.shape[0] == 0 or coords_other.shape[0] == 0:
        return np.zeros(dots.shape[0], dtype=bool)
    Ro = radii_other + probe
    tree = cKDTree(coords_other)
    k = min(6, coords_other.shape[0])
    d, idx = tree.query(dots, k=k)
    if d.ndim == 1:
        d, idx = d[:, None], idx[:, None]
    return (d < Ro[idx]).any(1)


# --------------------------------------------------------------------------- #
# pure-numpy 3D Zernike descriptor (order 20 -> 121 rotation invariants)
# --------------------------------------------------------------------------- #


def _zernike_index(order: int = DEFAULT_ORDER) -> list[tuple[int, int]]:
    """Enumerate ``(n, l)`` with ``0<=l<=n``, ``(n-l)`` even. order 20 -> 121 pairs."""
    out: list[tuple[int, int]] = []
    for n in range(order + 1):
        for l in range(n, -1, -1):
            if (n - l) % 2 == 0:
                out.append((n, l))
    return out


def _radial_norms(order: int, n_quad: int = 4000) -> dict[tuple[int, int], float]:
    """Per-``(n,l)`` constant making ``∫_0^1 R_nl(r)^2 r^2 dr = 1`` (midpoint rule)."""
    r = (np.arange(n_quad) + 0.5) / n_quad
    dr = 1.0 / n_quad
    norms: dict[tuple[int, int], float] = {}
    for n, l in _zernike_index(order):
        k = (n - l) // 2
        R = (r**l) * eval_jacobi(k, 0.0, l + 0.5, 2.0 * r * r - 1.0)
        integ = np.sum(R * R * r * r) * dr
        norms[(n, l)] = 1.0 / np.sqrt(integ) if integ > 0 else 0.0
    return norms


_NORM_CACHE: dict[int, dict[tuple[int, int], float]] = {}


def _norms_for(order: int) -> dict[tuple[int, int], float]:
    if order not in _NORM_CACHE:
        _NORM_CACHE[order] = _radial_norms(order)
    return _NORM_CACHE[order]


def _scale_to_unit_ball(points: np.ndarray, fill: float = 0.90) -> np.ndarray:
    """Center at centroid, scale so the 99.5th-percentile radius maps to ``fill``."""
    if points.shape[0] == 0:
        return points
    p = points - points.mean(0)
    rad = np.sqrt((p * p).sum(1))
    rmax = np.quantile(rad, 0.995)
    if rmax <= 0:
        rmax = 1.0
    return p * (fill / rmax)


def _zernike_invariants(points: np.ndarray, order: int = DEFAULT_ORDER) -> np.ndarray:
    """121-dim (order 20) rotation-invariant 3DZD of a pre-scaled point cloud.

    Points are assumed scaled into the unit ball (:func:`_scale_to_unit_ball`); any
    dot with ``r > 1`` is clipped out. Returns zeros if fewer than 4 dots survive.
    """
    idx = _zernike_index(order)
    F = np.zeros(len(idx), dtype=np.float64)
    if points.shape[0] < 4:
        return F
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    keep = (r <= 1.0) & (r > 1e-9)
    if keep.sum() < 4:
        return F
    x, y, z, r = x[keep], y[keep], z[keep], r[keep]
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # polar
    phi = np.arctan2(y, x)  # azimuth
    norms = _norms_for(order)

    by_l: dict[int, list[int]] = {}
    for pos, (n, l) in enumerate(idx):
        by_l.setdefault(l, []).append(pos)

    for l, positions in by_l.items():
        Y = np.stack([sph_harm_y(l, m, theta, phi) for m in range(-l, l + 1)], axis=0)
        Yc = np.conjugate(Y)
        for pos in positions:
            n, _l = idx[pos]
            k = (n - l) // 2
            R = norms[(n, l)] * (r**l) * eval_jacobi(k, 0.0, l + 0.5, 2.0 * r * r - 1.0)
            omega = Yc @ R
            F[pos] = np.sqrt(np.real(np.sum(omega * np.conjugate(omega))))
    return F


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na == 0 or nb == 0:
        return float("nan")
    return float((a * b).sum() / (na * nb))


# --------------------------------------------------------------------------- #
# reward entrypoint
# --------------------------------------------------------------------------- #


def _chain_atom_cloud(coords_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ``(n, 3, 3)`` backbone into ``(3n, 3)`` atoms + matching vdw radii."""
    n = coords_sub.shape[0]
    xyz = coords_sub.reshape(n * 3, 3)
    radii = np.tile(_BB_RADII, n)
    return xyz, radii


def _sc_core(
    a_xyz: np.ndarray,
    a_rad: np.ndarray,
    b_xyz: np.ndarray,
    b_rad: np.ndarray,
    *,
    order: int = DEFAULT_ORDER,
    nsphere: int = DEFAULT_NSPHERE,
    probe: float = PROBE,
) -> tuple[float, dict]:
    """Shared SC pipeline on two atom clouds -> ``(sc_raw, diag)``.

    ``a`` = antigen (chain A) cloud, ``b`` = binder (chain B) cloud, each ``(N, 3)``
    heavy-atom coordinates with matching ``(N,)`` van der Waals radii. Builds the
    crop-safe ΔSASA contact patches, reduces each to an order-``order`` 3DZD, and
    returns the **raw** Pearson correlation of the two descriptors (may be NaN for a
    degenerate/empty patch — callers clip/floor it) plus ``diag`` with the patch sizes.
    This is the single implementation shared by the backbone and all-atom entrypoints.
    """
    diag = {"sc": float("nan"), "n_patch_a": 0, "n_patch_b": 0}
    if a_xyz.shape[0] == 0 or b_xyz.shape[0] == 0:
        return float("nan"), diag

    dots_a = _accessible_dots(a_xyz, a_rad, probe, nsphere)
    dots_b = _accessible_dots(b_xyz, b_rad, probe, nsphere)
    patch_a = dots_a[_buried_by(dots_a, b_xyz, b_rad, probe)]
    patch_b = dots_b[_buried_by(dots_b, a_xyz, a_rad, probe)]
    diag["n_patch_a"] = int(patch_a.shape[0])
    diag["n_patch_b"] = int(patch_b.shape[0])

    Fa = _zernike_invariants(_scale_to_unit_ball(patch_a), order)
    Fb = _zernike_invariants(_scale_to_unit_ball(patch_b), order)
    sc = _pearson(Fa, Fb)
    diag["sc"] = sc
    return sc, diag


def shape_complementarity_reward_atoms(
    a_xyz,
    a_rad,
    b_xyz,
    b_rad,
    *,
    order: int = DEFAULT_ORDER,
    nsphere: int = DEFAULT_NSPHERE,
    probe: float = PROBE,
) -> tuple[float, dict]:
    """All-atom 3DZD shape-complementarity reward for one design.

    The full-atom analogue of :func:`shape_complementarity_reward`. Instead of a
    decoded ``(L, 3, 3)`` backbone it takes the antigen and binder **heavy-atom
    clouds directly** — ``(N_a, 3)`` / ``(N_b, 3)`` coordinates with matching
    ``(N_a,)`` / ``(N_b,)`` van der Waals radii (see :func:`radii_from_elements`).
    This is the reward-time entry once side chains are available (e.g. from a
    LigandMPNN repack), and it reproduces the offline full-atom SC validation
    (``scripts/_zernike_sc.py::sc_numpy``) since it shares the same numeric core.

    Parameters
    ----------
    a_xyz, b_xyz : array-like
        ``(N_a, 3)`` / ``(N_b, 3)`` antigen / binder heavy-atom coordinates.
    a_rad, b_rad : array-like
        ``(N_a,)`` / ``(N_b,)`` per-atom van der Waals radii.
    order, nsphere, probe
        3DZD order, Shrake-Rupley dots/atom, water probe radius — see
        :func:`shape_complementarity_reward`.

    Returns
    -------
    tuple[float, dict]
        ``(term, diag)`` where ``term = clip(pearson(F_a, F_b), 0, 1) ∈ [0, 1]``
        (NaN patch ⇒ 0) and ``diag = {"sc", "n_patch_a", "n_patch_b"}`` (``sc`` is
        the raw, unclipped Pearson).
    """
    a_xyz = np.asarray(a_xyz, dtype=np.float64)
    b_xyz = np.asarray(b_xyz, dtype=np.float64)
    a_rad = np.asarray(a_rad, dtype=np.float64)
    b_rad = np.asarray(b_rad, dtype=np.float64)
    sc, diag = _sc_core(a_xyz, a_rad, b_xyz, b_rad, order=order, nsphere=nsphere, probe=probe)
    if not np.isfinite(sc):
        return 0.0, diag
    return float(np.clip(sc, 0.0, 1.0)), diag


def shape_complementarity_reward(
    coords_full: np.ndarray,
    valid_mask,
    binder_mask,
    *,
    order: int = DEFAULT_ORDER,
    nsphere: int = DEFAULT_NSPHERE,
    probe: float = PROBE,
) -> tuple[float, dict]:
    """3DZD interface shape-complementarity reward for one generated design.

    .. note::

       Backbone-only entry (3 atoms/residue). This is the validated-negative path
       (AUROC 0.533); use :func:`shape_complementarity_reward_atoms` with packed
       side chains for the discriminative full-atom signal (AUROC ~0.65).

    Parameters
    ----------
    coords_full : np.ndarray
        ``(L, 3, 3)`` generated backbone (full padded length) in ``[N, CA, C]``
        order (``LeFlurGRPOTrainer._decode_backbone_coords`` output for one design).
    valid_mask, binder_mask : array-like of bool
        ``(L,)`` valid (non-pad) and designed-binder positions. The antigen is the
        valid non-binder positions.
    order : int
        3DZD order (20 -> 121 invariants).
    nsphere : int
        Shrake-Rupley dots per atom (surface resolution).
    probe : float
        Water probe radius (Å) for the accessible-surface + burial criterion.

    Returns
    -------
    tuple[float, dict]
        ``(term, diag)`` where ``term = clip(pearson(F_a, F_b), 0, 1) ∈ [0, 1]``
        (unweighted; NaN patch ⇒ 0) and ``diag = {"sc", "n_patch_a", "n_patch_b"}``.

    Notes
    -----
    A design with no valid binder or antigen residue, or with a degenerate contact
    patch (< 4 surviving dots), yields ``sc = NaN`` ⇒ ``term = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> ag = rng.normal(size=(30, 3, 3)) * 3.0
    >>> bd = ag + np.array([10.0, 0.0, 0.0])          # displaced copy -> in contact
    >>> coords = np.concatenate([ag, bd], axis=0)
    >>> valid = np.ones(60, dtype=bool)
    >>> binder = np.array([False] * 30 + [True] * 30)
    >>> term, diag = shape_complementarity_reward(coords, valid, binder, nsphere=48)
    >>> 0.0 <= term <= 1.0
    True
    """
    coords_full = np.asarray(coords_full, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    binder_mask = np.asarray(binder_mask, dtype=bool)

    bpos = np.nonzero(valid_mask & binder_mask)[0]
    apos = np.nonzero(valid_mask & ~binder_mask)[0]
    if bpos.size == 0 or apos.size == 0:
        return 0.0, {"sc": float("nan"), "n_patch_a": 0, "n_patch_b": 0}

    a_xyz, a_rad = _chain_atom_cloud(coords_full[apos])
    b_xyz, b_rad = _chain_atom_cloud(coords_full[bpos])
    return shape_complementarity_reward_atoms(a_xyz, a_rad, b_xyz, b_rad, order=order, nsphere=nsphere, probe=probe)
