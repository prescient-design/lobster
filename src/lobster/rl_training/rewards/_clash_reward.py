"""Smooth steric-clash + interface-contact shaping reward for LeFlur GRPO.

The interface-distribution reward (:mod:`._distribution_reward`) matches the
*composition* of a binder interface but is blind to 3-D geometry, so the policy
can score well while producing backbones that sterically **clash** (atoms
overlapping / chains crossing). The complementary failure mode is a binder that
avoids clashes by drifting **out of contact** with the antigen (a floating
binder). This module scores a single **smooth, well-behaved** ``[0, 1]`` term
that penalizes BOTH:

* **clash** — overlap of non-bonded heavy atoms (binder↔antigen and non-local
  binder↔binder), and
* **no contact** — absence of any binder residue near the antigen.

It mirrors :mod:`._distribution_reward`: pure numpy (no torch / Protenix / trl),
unit-testable in isolation, and inert (contributes exactly 0) when its weight is
0. Like the other terms it is used as a **scalar** in GRPO (advantage-weighted
log-prob) — it is *not* back-propagated — so it need not be autograd
differentiable, only numerically smooth / well-behaved.

Design → reward
---------------
For one design with decoded backbone ``coords_full (L, 3, 3)`` in ``[N, CA, C]``
order plus ``valid_mask`` / ``binder_mask`` ``(L,)``:

**Atom cloud.** Per valid residue use heavy atoms N, CA, C **+ Cβ** (Cβ via the
standard bond-geometry formula, replicated from
:func:`lobster.model.latent_generator.utils.mini3di._encoder.calculate_cb`).
Glycine gets a virtual Cβ — fine for sterics.

**Clash score** ``∈ (0, 1]`` (1 = clash-free), a smooth soft-core over
non-bonded atom pairs (binder×antigen, plus binder×binder with residue
separation ``|i − j| ≥ seq_sep``):

    p(d)      = 0.5·(1 − tanh((d − d_clash) / clash_soft))     # C¹-continuous
    E_clash   = Σ p(d)
    clash     = exp(−E_clash / clash_scale)

``p`` is a smooth sigmoid transition centred at ``d_clash`` (→1 for ``d ≪
d_clash``, →0 for ``d ≫ d_clash``) with **no hard cutoff / clip corner**; distant
pairs contribute ≈0 so the sum is dominated by genuine overlaps. ``clash`` is
bounded, saturating, and can never go negative.

**Contact score** ``∈ [0, 1]`` (0 = floating *or* over-large interface), a smooth
band on the *binder interface fraction* — the fraction of binder residues in
contact with the antigen:

    d_min_i        = min CA–CA distance of binder residue i to any valid antigen
    soft_n_iface   = Σ_i sigmoid((contact_d0 − d_min_i) / contact_soft)  # smooth count
    iface_frac     = soft_n_iface / n_binder
    contact        = iface_frac_band(iface_frac; frac_lo, frac_peak, frac_hi)

``iface_frac_band`` is an **asymmetric raised-cosine bump** that is 0 outside
``[frac_lo, frac_hi]``, peaks at 1.0 at ``frac_peak``, and is ``C¹``-continuous
everywhere (zero slope at all three knots — no hard cutoff / corner). This targets
the *native* interface-fraction band measured on passing Proteina-Complexa binders
(median ≈ 0.16, bulk 0.07–0.34): it **penalizes no contact** (``iface_frac <
frac_lo`` — a floating binder) AND **penalizes an over-large / interpenetrating
interface** (``iface_frac > frac_hi`` — the ~0.63 fraction seen in the clashing
diverged GRPO rollouts), while rewarding the healthy band around ``frac_peak``.

**Combined term** ``= clash · contact`` ``∈ [0, 1]`` — maximised only by a clean
backbone whose interface fraction sits in the native band; each failure mode
(clash, floating, over-large) drives it toward 0. The trainer multiplies by
``w_clash_contact``.

Notes
-----
Clash is backbone+Cβ only (no sidechains are decoded), so it catches backbone
crashes and chain crossings, not sidechain rotamer clashes. Decoded backbones are
codec reconstructions (~1–2.7 Å RMSD), so the defaults (``d_clash=3.0``,
``clash_soft=0.5``) are deliberately tolerant of reconstruction noise.
"""

from __future__ import annotations

import numpy as np

# Cβ bond-geometry constants (mirror mini3di._encoder.calculate_cb).
_CB_A = -0.58273431
_CB_B = 0.56802827
_CB_C = -0.54067466

# Defaults (Å where a distance; see module docstring for the reward shapes).
D_CLASH = 2.2  # soft-core clash center: heavy atoms closer than this overlap
CLASH_SOFT = 0.5  # clash transition width
CLASH_SCALE = 50.0  # E_clash -> clash_score saturation scale
CONTACT_D0 = 8.0  # interface band center (matches IFACE_THRESH everywhere)
CONTACT_SOFT = 1.0  # contact transition width (Å) for the soft interface count
# Native interface-fraction band (fraction of binder residues at the interface).
# Calibrated to passing Proteina-Complexa binders (n=1008, generated backbones):
# median ~0.16, bulk 0.07-0.34. Peak sits on the passing median.
FRAC_LO = 0.05  # below this the binder is under-contacting (floating) -> 0
FRAC_PEAK = 0.16  # healthy native interface fraction -> peak reward 1.0
FRAC_HI = 0.4  # above this the interface is over-large / interpenetrating -> 0
SEQ_SEP = 2  # exclude binder pairs within this residue separation (self + bonded)


def _backbone_cb(coords: np.ndarray) -> np.ndarray:
    """Virtual Cβ per residue from ``(n, 3, 3)`` ``[N, CA, C]`` backbone coords.

    Parameters
    ----------
    coords : np.ndarray
        ``(n, 3, 3)`` backbone coordinates in ``[N, CA, C]`` order.

    Returns
    -------
    np.ndarray
        ``(n, 3)`` Cβ coordinates via ``Cb = A·(b×c) + B·b + C·c + CA`` where
        ``b = CA − N`` and ``c = C − CA``.
    """
    n = coords[:, 0, :]
    ca = coords[:, 1, :]
    c = coords[:, 2, :]
    b = ca - n
    cc = c - ca
    a = np.cross(b, cc)
    return _CB_A * a + _CB_B * b + _CB_C * cc + ca


def _atom_cloud(coords_sub: np.ndarray, include_cb: bool) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-residue backbone (+Cβ) atoms into an ``(m, 3)`` cloud.

    Parameters
    ----------
    coords_sub : np.ndarray
        ``(n, 3, 3)`` backbone coordinates in ``[N, CA, C]`` order.
    include_cb : bool
        Append a virtual Cβ atom per residue.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(xyz (m, 3), res_idx (m,))`` where ``res_idx`` maps each atom back to
        its residue (``0..n-1``); ``m = n·3`` (or ``n·4`` with Cβ).
    """
    n = coords_sub.shape[0]
    blocks = [coords_sub[:, 0, :], coords_sub[:, 1, :], coords_sub[:, 2, :]]
    if include_cb:
        blocks.append(_backbone_cb(coords_sub))
    xyz = np.concatenate(blocks, axis=0)
    res_idx = np.tile(np.arange(n), len(blocks))
    return xyz, res_idx


def _pdist2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between ``(m, 3)`` and ``(k, 3)`` clouds."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def _softcore(d: np.ndarray, d_clash: float, clash_soft: float) -> np.ndarray:
    """Smooth per-pair clash penalty ``0.5·(1 − tanh((d − d_clash)/soft)) ∈ (0, 1)``."""
    return 0.5 * (1.0 - np.tanh((d - d_clash) / clash_soft))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid (no overflow for large ``|x|``)."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _iface_frac_band(frac, lo: float, peak: float, hi: float):
    """Smooth asymmetric raised-cosine bump on the interface fraction.

    A ``C¹``-continuous window that is 0 for ``frac ≤ lo`` and ``frac ≥ hi``, rises
    to 1.0 at ``frac = peak``, and has **zero slope at all three knots** (``lo``,
    ``peak``, ``hi``) — no hard cutoff, no corner. Built from two half-cosines of
    independent width (``peak − lo`` on the rising side, ``hi − peak`` on the
    falling side), so the two shoulders can be asymmetric while the join at
    ``peak`` stays smooth.

    Parameters
    ----------
    frac : float or np.ndarray
        Interface fraction(s) in ``[0, 1]``.
    lo, peak, hi : float
        Lower zero-crossing, reward peak, and upper zero-crossing
        (``lo < peak < hi``).

    Returns
    -------
    float or np.ndarray
        Band value(s) in ``[0, 1]`` (scalar in ⇒ scalar out).

    Examples
    --------
    >>> import numpy as np
    >>> float(_iface_frac_band(0.2, 0.1, 0.2, 0.4))   # peak
    1.0
    >>> float(_iface_frac_band(0.1, 0.1, 0.2, 0.4))   # lower edge
    0.0
    >>> float(_iface_frac_band(0.4, 0.1, 0.2, 0.4))   # upper edge
    0.0
    >>> float(_iface_frac_band(0.63, 0.1, 0.2, 0.4))  # over-large interface
    0.0
    """
    x = np.asarray(frac, dtype=np.float64)
    scalar = x.ndim == 0
    f = np.atleast_1d(x)
    out = np.zeros_like(f)
    left = (f > lo) & (f < peak)
    right = (f >= peak) & (f < hi)
    out[left] = 0.5 * (1.0 - np.cos(np.pi * (f[left] - lo) / (peak - lo)))
    out[right] = 0.5 * (1.0 + np.cos(np.pi * (f[right] - peak) / (hi - peak)))
    return float(out[0]) if scalar else out


def clash_contact_reward(
    coords_full: np.ndarray,
    valid_mask,
    binder_mask,
    *,
    d_clash: float = D_CLASH,
    clash_soft: float = CLASH_SOFT,
    clash_scale: float = CLASH_SCALE,
    contact_d0: float = CONTACT_D0,
    contact_soft: float = CONTACT_SOFT,
    frac_lo: float = FRAC_LO,
    frac_peak: float = FRAC_PEAK,
    frac_hi: float = FRAC_HI,
    seq_sep: int = SEQ_SEP,
    include_cb: bool = True,
    return_eres: bool = False,
) -> tuple[float, dict]:
    """Smooth steric-clash + interface-contact reward for one generated design.

    Parameters
    ----------
    coords_full : np.ndarray
        ``(L, 3, 3)`` generated backbone (full padded length) in ``[N, CA, C]``
        order (``LeFlurGRPOTrainer._decode_backbone_coords`` output for one
        design).
    valid_mask, binder_mask : array-like of bool
        ``(L,)`` valid (non-pad) and designed-binder positions.
    d_clash, clash_soft, clash_scale : float
        Soft-core clash center (Å), transition width (Å), and ``E_clash →
        clash_score`` saturation scale.
    contact_d0, contact_soft : float
        Interface-band center (Å) and transition width (Å) for the smooth count
        of binder residues in contact with the antigen.
    frac_lo, frac_peak, frac_hi : float
        Interface-fraction band: ``contact_score = 0`` at/below ``frac_lo`` and
        at/above ``frac_hi``, peaking at 1.0 at ``frac_peak`` (see
        :func:`_iface_frac_band`). Defaults target the native passing-Complexa
        band (0.05 / 0.16 / 0.4).
    seq_sep : int
        Binder residue pairs with ``|i − j| < seq_sep`` are excluded from the
        clash sum (self + peptide-bonded neighbours).
    include_cb : bool
        Include a virtual Cβ atom per residue in the clash atom cloud.
    return_eres : bool
        When ``True`` add ``diag["E_clash_res"]`` — the per-binder-residue clash
        energy of shape ``(n_valid_binder,)`` (in the order of the valid binder
        positions, ``np.nonzero(valid_mask & binder_mask)``), whose sum equals
        ``E_clash`` **exactly**. Each binder↔antigen pair is attributed fully to
        its binder residue; each non-local binder↔binder pair is split 50/50 between
        its two endpoint residues. Used by the per-token clash advantage to assign
        per-residue credit to the structure track. Default ``False`` (byte-identical:
        no extra key, and the scalar ``E_clash`` is unchanged).

    Returns
    -------
    tuple[float, dict]
        ``(term, diag)`` where ``term = clash_score · contact_score ∈ [0, 1]``
        (unweighted) and ``diag = {"clash_score", "contact_score", "E_clash",
        "soft_n_iface", "iface_frac"}`` (plus ``"E_clash_res"`` when
        ``return_eres``).

    Notes
    -----
    A design with no valid binder or no valid antigen residue has no interface, so
    ``iface_frac = 0`` ⇒ ``contact_score = 0`` ⇒ ``term = 0``; ``clash_score`` still
    reflects any binder-internal overlap. All components are continuous in the
    coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> # two well-separated 3-residue chains: clash-free, in contact
    >>> ag = np.stack([np.array([[0, 0, z], [0, 1, z], [0, 2, z]]) for z in (0, 3, 6)])
    >>> bd = ag + np.array([6.0, 0.0, 0.0])
    >>> coords = np.concatenate([ag, bd], axis=0).astype(float)
    >>> valid = np.ones(6, dtype=bool)
    >>> binder = np.array([False, False, False, True, True, True])
    >>> term, diag = clash_contact_reward(coords, valid, binder)
    >>> 0.0 <= term <= 1.0
    True
    """
    coords_full = np.asarray(coords_full, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    binder_mask = np.asarray(binder_mask, dtype=bool)

    bpos = np.nonzero(valid_mask & binder_mask)[0]
    apos = np.nonzero(valid_mask & ~binder_mask)[0]

    # ------------------------------------------------------------- clash score
    e_clash = 0.0
    # Per-binder-residue clash energy (local indexing 0..nb-1). Populated only
    # when requested; sum == e_clash exactly.
    e_res = np.zeros(bpos.size, dtype=np.float64) if return_eres else None
    if bpos.size:
        b_xyz, b_res = _atom_cloud(coords_full[bpos], include_cb)
        # binder x antigen: every cross-chain heavy-atom pair.
        if apos.size:
            a_xyz, _ = _atom_cloud(coords_full[apos], include_cb)
            p_ba = _softcore(_pdist2(b_xyz, a_xyz), d_clash, clash_soft)  # (mb, ma)
            e_clash += float(p_ba.sum())
            if return_eres:
                # Attribute each binder-atom's antigen penalty fully to its residue.
                np.add.at(e_res, b_res, p_ba.sum(axis=1))
        # binder x binder: non-local pairs only, each counted once (i < j).
        if b_xyz.shape[0] > 1:
            dbb = _pdist2(b_xyz, b_xyz)
            sep_ok = np.abs(b_res[:, None] - b_res[None, :]) >= seq_sep
            upper = np.triu(np.ones_like(sep_ok, dtype=bool), k=1)
            mask = sep_ok & upper
            if mask.any():
                p_bb = _softcore(dbb[mask], d_clash, clash_soft)
                e_clash += float(p_bb.sum())
                if return_eres:
                    # Split each intra-binder pair 50/50 between its endpoints.
                    ii, jj = np.nonzero(mask)
                    np.add.at(e_res, b_res[ii], 0.5 * p_bb)
                    np.add.at(e_res, b_res[jj], 0.5 * p_bb)
    clash_score = float(np.exp(-e_clash / clash_scale))

    # ----------------------------------------------------------- contact score
    # Smooth count of binder residues at the interface -> interface fraction ->
    # asymmetric raised-cosine band centred on the native passing fraction.
    soft_n_iface = 0.0
    iface_frac = 0.0
    if bpos.size and apos.size:
        ca = coords_full[:, 1, :]  # CA is atom index 1
        d = _pdist2(ca[bpos], ca[apos])  # (nvb, na)
        d_min = d.min(axis=1)
        soft_n_iface = float(_sigmoid((contact_d0 - d_min) / contact_soft).sum())
        iface_frac = soft_n_iface / bpos.size
    contact_score = float(_iface_frac_band(iface_frac, frac_lo, frac_peak, frac_hi))

    term = clash_score * contact_score
    diag = {
        "clash_score": clash_score,
        "contact_score": contact_score,
        "E_clash": e_clash,
        "soft_n_iface": soft_n_iface,
        "iface_frac": iface_frac,
    }
    if return_eres:
        diag["E_clash_res"] = e_res
    return term, diag
