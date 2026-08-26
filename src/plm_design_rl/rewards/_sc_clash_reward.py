"""All-atom side-chain steric-clash shaping reward for LeFlur GRPO (whole-binder).

The wired :mod:`._clash_reward` term clashes only backbone + a *virtual* Cβ, so any
steric overlap that appears only once real side chains are placed is invisible to it.
This module scores the same physical property — **atoms should not overlap** — on the
**full-atom** side-chain cloud produced by the LigandMPNN packer (:mod:`lobster.model.
latent_generator.io._pack_pdb`), served by the CPU repack worker pool. It is the reward
counterpart of the offline engine ``scripts/_sc_clash.py``.

Scope — the reward is over the **whole binder**
-----------------------------------------------
Two clash sources both count as physical badness, and both are summed over the *entire*
binder (not just its interface residues):

* **binder self-clash** — every non-local intra-binder heavy-atom pair
  (``|i − j| > seq_sep`` residues apart). A binder fold that collides with itself is
  non-physical regardless of how it docks. Computed over all binder residues.
* **binder↔antigen clash** — every binder heavy atom against every antigen heavy atom.
  This is the docking-quality overlap (does the binder interpenetrate the target once
  side chains are real?), scored over *all* binder atoms, not an interface crop.

Interface-restricted counts (clash confined to residues near the antigen) are computed
and returned **only as diagnostics** — they never enter the reward. This follows the
design directive: *track the interface, but reward over the binder* — because a clash
anywhere in the binder is biophysically bad, and cropping the signal to the interface
would let self-clashing folds through.

Biophysical grounding (not a correlation-chased term)
-----------------------------------------------------
This term is a hard physical constraint, not a learned pass-rate predictor: overlapping
van-der-Waals spheres are energetically forbidden, full stop. It is intentionally kept
even though offline pass-rate correlations for such geometry terms are weak — the point
is to keep the policy inside the physically valid manifold, not to chase a proxy metric.
Default weight is 0 (opt-in), and like every other term it contributes exactly 0 when its
weight is 0.

Design → reward
---------------
The worker repacks each design and calls :func:`binder_clash_terms` on the packed atom14
clouds, returning a metrics dict; the policy side maps it to a scalar with
:func:`sc_clash_reward`::

    E_clash   = E_clash_binder + E_clash_iface          # whole-binder physical badness
    reward    = exp(−E_clash / sc_clash_scale)  ∈ (0, 1]  (1 = clash-free)

Soft-core: for a heavy-atom pair at distance ``d`` with Bondi VDW radii ``r_i, r_j`` the
overlap is ``o = (r_i + r_j) − d``; the pair contributes ``sigmoid((o − tol)/sigma)`` — a
smooth, bounded (0..1) "is this pair clashing" indicator (``tol`` allows a small VDW
compression before it counts). ``E_clash`` is the soft sum over pairs. The mapper is
bounded, saturating, and can never go negative, so a floored/missing design is 0.

Like the other reward terms this module is pure numpy (no torch / trl); the packer is
imported lazily only where a cloud is actually built. Radii are Bondi VDW, numerically
identical to :data:`._shape_reward.VDW_RADII` / ``scripts/_zernike_sc.py`` /
``scripts/_sc_clash.py`` so the reward reproduces the offline SC-clash atom model.
"""

from __future__ import annotations

import numpy as np

# Bondi VDW radii — kept identical to rewards/_shape_reward.VDW_RADII and scripts/_sc_clash.py.
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
DEFAULT_SIGMA = 0.5  # softness of the clash sigmoid (Å)
DEFAULT_TOL = 0.4  # allowed VDW compression before a pair counts as a clash (Å)
SEQ_SEP = 2  # exclude intra-binder pairs within this residue separation (self + bonded)
# E_clash -> reward saturation scale. E_clash is all-atom (much larger than the
# backbone+Cβ _clash_reward), so this is set larger than that term's CLASH_SCALE=50.
# Calibrate against scripts/_sc_clash_analyze.py distributions before turning the weight on.
SC_CLASH_SCALE = 150.0
# Density-mode saturation scale (per-residue energy, retraction-resistant). Calibrated on
# base vs grpo_step150 packed complexes (scripts/_calib_scclash_density.py): base density
# ~12.1 -> exp(-12.1/17.5) ~ 0.50 starts the term mid-range with headroom both ways.
SC_CLASH_DENSITY_SCALE = 17.5
IFACE_D0 = 8.0  # CA-CA cutoff (Å) marking a binder residue as "at the interface" (diagnostic)


def radii_from_elements(elements) -> np.ndarray:
    """Per-atom Bondi VDW radii for an element list (unknown element → carbon, 1.70 Å)."""
    return np.array([VDW_RADII.get(str(e).strip().upper(), 1.70) for e in elements], dtype=np.float64)


def _pair_clash(
    xyz_a: np.ndarray,
    rad_a: np.ndarray,
    xyz_b: np.ndarray,
    rad_b: np.ndarray,
    sigma: float,
    tol: float,
    block: int = 2048,
) -> tuple[float, int]:
    """Soft clash ``E`` and hard clash-pair count over the full cross set ``a × b``.

    Blocked over ``a`` to bound memory for large clouds. Returns ``(E, n_hard)`` where
    ``E = Σ sigmoid((overlap − tol)/sigma)`` and ``n_hard`` counts pairs with
    ``overlap > tol``.
    """
    if len(xyz_a) == 0 or len(xyz_b) == 0:
        return 0.0, 0
    E = 0.0
    n_hard = 0
    rsum_b = rad_b[None, :]
    for s in range(0, len(xyz_a), block):
        xa = xyz_a[s : s + block]
        ra = rad_a[s : s + block][:, None]
        d = np.sqrt(np.maximum(((xa[:, None, :] - xyz_b[None, :, :]) ** 2).sum(-1), 1e-12))
        overlap = (ra + rsum_b) - d  # >0 => VDW spheres overlap
        E += float((1.0 / (1.0 + np.exp(-(overlap - tol) / sigma))).sum())
        n_hard += int((overlap > tol).sum())
    return E, n_hard


def interface_allatom_clash(
    ag_xyz: np.ndarray,
    ag_elem,
    bd_xyz: np.ndarray,
    bd_elem,
    n_iface_res: int = 0,
    sigma: float = DEFAULT_SIGMA,
    tol: float = DEFAULT_TOL,
) -> dict:
    """Cross-chain antigen×binder heavy-atom clash (docking-quality, over all binder atoms)."""
    ra = radii_from_elements(ag_elem)
    rb = radii_from_elements(bd_elem)
    E, n_hard = _pair_clash(ag_xyz, ra, bd_xyz, rb, sigma, tol)
    norm = float(n_iface_res) if n_iface_res > 0 else float(max(len(bd_xyz), 1))
    return {
        "E_clash_iface": E,
        "e_clash_iface_norm": E / norm,
        "n_clash_atoms_iface": n_hard,
        "n_ag_atoms": len(ag_xyz),
        "n_bd_atoms": len(bd_xyz),
    }


def binder_selfclash(
    bd_xyz: np.ndarray,
    bd_elem,
    res_idx: np.ndarray,
    seq_sep: int = SEQ_SEP,
    sigma: float = DEFAULT_SIGMA,
    tol: float = DEFAULT_TOL,
    block: int = 2048,
) -> dict:
    """Intra-binder heavy-atom self-clash for residue pairs > ``seq_sep`` apart (whole binder).

    Excludes same-residue (bonded) and near-sequence neighbours, whose backbone geometry
    trivially "overlaps"; only non-local self-clash signals a non-physical fold. Each pair
    is counted once (global upper triangular).
    """
    n = len(bd_xyz)
    if n == 0:
        return {"E_clash_binder": 0.0, "e_clash_binder_norm": 0.0, "n_clash_atoms_binder": 0, "n_bd_atoms": 0}
    rad = radii_from_elements(bd_elem)
    res_idx = np.asarray(res_idx)
    E = 0.0
    n_hard = 0
    for s in range(0, n, block):
        e = min(s + block, n)
        xa = bd_xyz[s:e]
        ra = rad[s:e][:, None]
        ria = res_idx[s:e][:, None]
        d = np.sqrt(np.maximum(((xa[:, None, :] - bd_xyz[None, :, :]) ** 2).sum(-1), 1e-12))
        overlap = (ra + rad[None, :]) - d
        far = np.abs(ria - res_idx[None, :]) > seq_sep  # non-local residue pairs only
        gi = np.arange(s, e)[:, None]  # global upper-triangular => each pair counted once
        gj = np.arange(n)[None, :]
        keep = far & (gj > gi)
        ov = np.where(keep, overlap, -np.inf)
        E += float((1.0 / (1.0 + np.exp(-(ov - tol) / sigma)))[keep].sum())
        n_hard += int((ov > tol).sum())
    n_res = int(len(np.unique(res_idx)))
    return {
        "E_clash_binder": E,
        "e_clash_binder_norm": E / max(n_res, 1),
        "n_clash_atoms_binder": n_hard,
        "n_bd_atoms": n,
    }


def cloud_from_atom14(X14: np.ndarray, X_m: np.ndarray, S: np.ndarray):
    """LigandMPNN atom14 block -> ``(xyz (M,3), elements list, res_idx (M,))``.

    Imports the atom14→element map from the packer engine
    (:mod:`lobster.model.latent_generator.io._pack_pdb`), a numpy-only import.

    Parameters
    ----------
    X14 : np.ndarray
        ``(L, A, 3)`` packed atom14 coordinates (``A`` = 14 for full atom14, or a
        backbone+Cβ slice).
    X_m : np.ndarray
        ``(L, A)`` valid-atom mask (1 where an atom is present).
    S : np.ndarray
        ``(L,)`` residue integer identities (restype 0..19, 20 = UNK).

    Returns
    -------
    tuple[np.ndarray, list[str], np.ndarray]
        Heavy-atom ``xyz (M, 3)``, per-atom element symbols, and per-atom residue index
        (``0..L-1``). Residues with no valid atoms are dropped.
    """
    from lobster.model.latent_generator.io._pack_pdb import _AA3_BY_INT, _RESTYPE_ATOM14_ELEMENTS

    n_atoms = X14.shape[1]  # 14 for full atom14; smaller for a backbone slice
    xyz_list, elem_list, ridx_list = [], [], []
    for i in range(X14.shape[0]):
        aa3 = _AA3_BY_INT.get(int(S[i]), "UNK")
        elems = _RESTYPE_ATOM14_ELEMENTS[aa3]  # full-14 element order; slice matches atom14
        sel = X_m[i].astype(bool)
        if not sel.any():
            continue
        xyz_list.append(X14[i][sel])
        elem_list.extend([elems[j] for j in range(n_atoms) if sel[j]])
        ridx_list.extend([i] * int(sel.sum()))
    if not xyz_list:
        return np.empty((0, 3), dtype=np.float64), [], np.empty(0, dtype=np.int64)
    return (
        np.concatenate(xyz_list, axis=0).astype(np.float64),
        elem_list,
        np.array(ridx_list, dtype=np.int64),
    )


def binder_clash_terms(
    bd_X14: np.ndarray,
    bd_Xm: np.ndarray,
    bd_S: np.ndarray,
    ag_X14: np.ndarray,
    ag_Xm: np.ndarray,
    ag_S: np.ndarray,
    *,
    seq_sep: int = SEQ_SEP,
    sigma: float = DEFAULT_SIGMA,
    tol: float = DEFAULT_TOL,
    iface_d0: float = IFACE_D0,
) -> dict:
    """Whole-binder all-atom clash metrics from packed antigen + binder atom14 clouds.

    Computes the two reward sources — binder self-clash (over the whole binder) and
    binder↔antigen interface clash (over all binder atoms) — plus interface-restricted
    diagnostics. This is what the repack worker calls after packing; the scalar reward
    is :func:`sc_clash_reward` applied to the returned dict.

    Parameters
    ----------
    bd_X14, bd_Xm, bd_S : np.ndarray
        Packed binder atom14 coordinates ``(Lb, 14, 3)``, valid-atom mask ``(Lb, 14)``,
        and residue identities ``(Lb,)``.
    ag_X14, ag_Xm, ag_S : np.ndarray
        Packed antigen atom14 coordinates ``(La, 14, 3)``, mask ``(La, 14)``, residue
        identities ``(La,)``.
    seq_sep : int
        Intra-binder residue pairs with ``|i − j| ≤ seq_sep`` are excluded from the
        self-clash sum (self + peptide-bonded neighbours).
    sigma, tol : float
        Soft-core sigmoid width (Å) and allowed VDW compression (Å) before a pair counts.
    iface_d0 : float
        CA–CA cutoff (Å) marking a binder residue as "at the interface" — used only to
        compute the diagnostic interface-restricted clash (never the reward).

    Returns
    -------
    dict
        Reward inputs ``E_clash_binder`` (whole-binder self-clash) and ``E_clash_iface``
        (all binder atoms × antigen atoms); their sum ``E_clash_total`` and the mapped
        ``term`` (so a cached result already carries the scalar). Plus diagnostics:
        normalized per-residue energies, hard clash-atom counts, atom counts, and the
        interface-restricted ``E_clash_iface_res`` / ``n_iface_res`` (clash confined to
        binder residues within ``iface_d0`` of the antigen — tracked, not rewarded).

    Notes
    -----
    A design with no binder atoms yields all-zero energies → ``term = 1`` (clash-free by
    vacuity); the SC-shape / distribution terms handle the "floating / empty" failure
    mode, this term handles overlap only.
    """
    bd_xyz, bd_elem, bd_res = cloud_from_atom14(bd_X14, bd_Xm, bd_S)
    ag_xyz, ag_elem, _ = cloud_from_atom14(ag_X14, ag_Xm, ag_S)

    self_d = binder_selfclash(bd_xyz, bd_elem, bd_res, seq_sep=seq_sep, sigma=sigma, tol=tol)
    iface_d = interface_allatom_clash(ag_xyz, ag_elem, bd_xyz, bd_elem, sigma=sigma, tol=tol)

    # Diagnostic only: clash restricted to binder atoms whose residue is at the interface
    # (min CA-CA to antigen < iface_d0). Tracks "where" the interface clash sits; the
    # rewarded E_clash_iface above is over ALL binder atoms per the whole-binder directive.
    e_clash_iface_res = 0.0
    n_iface_res = 0
    if bd_xyz.size and ag_xyz.size:
        bd_ca = _ca_by_residue(bd_X14, bd_Xm)
        ag_ca = _ca_by_residue(ag_X14, ag_Xm)
        if bd_ca.size and ag_ca.size:
            d_min = np.sqrt(np.maximum(((bd_ca[:, None, :] - ag_ca[None, :, :]) ** 2).sum(-1), 1e-12)).min(axis=1)
            iface_res = np.nonzero(d_min < iface_d0)[0]
            n_iface_res = int(iface_res.size)
            if n_iface_res:
                sel_atoms = np.isin(bd_res, iface_res)
                e_clash_iface_res, _ = _pair_clash(
                    bd_xyz[sel_atoms],
                    radii_from_elements([bd_elem[k] for k in np.nonzero(sel_atoms)[0]]),
                    ag_xyz,
                    radii_from_elements(ag_elem),
                    sigma,
                    tol,
                )

    e_total = float(self_d["E_clash_binder"] + iface_d["E_clash_iface"])
    out = {
        # reward inputs (whole-binder scope)
        "E_clash_binder": self_d["E_clash_binder"],
        "E_clash_iface": iface_d["E_clash_iface"],
        "E_clash_total": e_total,
        "term": sc_clash_reward({"E_clash_total": e_total}),
        # diagnostics
        "e_clash_binder_norm": self_d["e_clash_binder_norm"],
        "e_clash_iface_norm": iface_d["e_clash_iface_norm"],
        "n_clash_atoms_binder": self_d["n_clash_atoms_binder"],
        "n_clash_atoms_iface": iface_d["n_clash_atoms_iface"],
        "n_bd_atoms": self_d["n_bd_atoms"],
        "n_ag_atoms": iface_d["n_ag_atoms"],
        "E_clash_iface_res": e_clash_iface_res,  # interface-restricted (tracked, not rewarded)
        "n_iface_res": n_iface_res,
    }
    return out


def _ca_by_residue(X14: np.ndarray, X_m: np.ndarray) -> np.ndarray:
    """Per-residue CA coordinates (atom14 index 1) for residues with a valid CA."""
    if X14.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    has_ca = X_m[:, 1].astype(bool)
    return X14[has_ca, 1, :].astype(np.float64)


def sc_clash_reward(
    res: dict | None,
    *,
    sc_clash_scale: float = SC_CLASH_SCALE,
    density: bool = False,
    sc_clash_density_scale: float = SC_CLASH_DENSITY_SCALE,
) -> float:
    """Scalar clash reward from a metrics dict: ``exp(−E / scale)`` ∈ (0, 1].

    Bounded, saturating, never negative. ``1.0`` = clash-free; larger overlap drives it
    toward 0. A missing/failed design (``None``) floors to 0.0.

    Two energy modes:

    - **absolute** (default) — ``E = E_clash_total`` (whole-binder self-clash + all
      binder↔antigen atom overlap). Retracting the binder off the antigen sheds interface
      atoms and lowers ``E`` *even with no genuine de-clashing*, so this mode rewards
      interface retraction — the cannibalization failure seen in the co-equal sc_clash+aar
      arm (dist, the pass-predictor, drifted away while sc_clash rose).
    - **density** (``density=True``) — ``E = E_clash_binder/n_res + E_clash_iface_res/n_iface_res``
      (per-residue self-clash + per-interface-residue clash). Both numerator and
      denominator shrink together under retraction, so a *smaller* interface no longer
      wins automatically — the policy must actually de-clash *retained* contacts to gain
      reward. This is the retraction-resistant form (see ``scripts/_calib_scclash_density.py``:
      among warm-start designs the one with the MOST interface residues has the LOWEST
      density, confirming density does not reward shrinking the interface).

    Parameters
    ----------
    res : dict | None
        Metrics dict from :func:`binder_clash_terms`. Absolute mode uses ``E_clash_total``
        (falls back to ``E_clash_binder + E_clash_iface``). Density mode uses
        ``e_clash_binder_norm`` (= ``E_clash_binder / n_res``) + ``E_clash_iface_res`` /
        ``n_iface_res``. ``None`` → 0.0.
    sc_clash_scale : float
        Absolute-mode saturation scale (Å-atom-pair units).
    density : bool
        Select the per-residue density energy (retraction-resistant) instead of absolute.
    sc_clash_density_scale : float
        Density-mode saturation scale. Calibrated so a clashy warm-start (base density
        ≈ 12) starts the term mid-range (~0.5): ``exp(−12/17.5) ≈ 0.5``.

    Returns
    -------
    float
        Reward in ``[0, 1]`` (``0.0`` if ``res`` is ``None`` or non-finite).
    """
    if res is None:
        return 0.0
    if density:
        binder_dens = float(res.get("e_clash_binder_norm", 0.0))
        n_if = int(res.get("n_iface_res", 0) or 0)
        e_if_res = float(res.get("E_clash_iface_res", 0.0) or 0.0)
        iface_dens = e_if_res / n_if if n_if > 0 else 0.0
        e = binder_dens + iface_dens
        scale = float(sc_clash_density_scale)
    else:
        e = res.get("E_clash_total")
        if e is None:
            e = float(res.get("E_clash_binder", 0.0)) + float(res.get("E_clash_iface", 0.0))
        e = float(e)
        scale = float(sc_clash_scale)
    if not np.isfinite(e):
        return 0.0
    # E is a sum of (per-residue) sigmoid overlaps (>= 0), so exp(-E/scale) is already in
    # (0, 1]; clamp explicitly to guarantee the reward stays in [0, 1] under any E path.
    return float(min(1.0, max(0.0, np.exp(-e / scale))))
