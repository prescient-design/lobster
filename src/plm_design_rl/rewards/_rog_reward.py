"""Radius-of-gyration compactness shaping reward for LeFlur GRPO.

Every other structural reward — steric clash (:mod:`._clash_reward`), interface
3Di/AA distribution (:mod:`._distribution_reward`), 3DZD shape complementarity
(:mod:`._shape_reward`), chain-break (:mod:`._chainbreak_reward`) — measures a
*local* property (a contact, a bond, a patch). None of them sees the binder's
**global fold state**: whether the chain is a compact globule or an over-extended
tangle. The cross-arm geometry analysis (Table 6 of the two-reward-sets proposal)
found this is a real, *length-independent* gap — at essentially matched chain length
the passing Proteina-Complexa references sit near the globular target while every
trained arm is measurably over-extended:

    arm             compactness (r0=2.2)   Rg / (r0 · N^(1/3))
    Complexa ref            0.76                 1.31
    six-term step275        0.67                 1.50
    base                    0.63                 1.58
    scalar-AAR              0.60                 1.67
    CHORD-SFT step175       0.54                 1.85   (most over-extended)

This module scores a single ``[0, 1]`` term that rewards a compact binder backbone
(higher = better) using the **compactness** form

    Rg_actual   = sqrt( mean_i ‖CA_i − mean(CA)‖² )     # binder Cα radius of gyration
    Rg_compact  = r0 · N^(1/3)                          # ideal globule Rg at N residues
    compactness = Rg_compact / Rg_actual                # ~0.7 globular, higher = more compact

``compactness`` is length-normalized by construction (the ``N^(1/3)`` compact-globule
scaling divides out chain length), so it targets fold state, not size. It equals the
mass-weighted ``biotite.structure.gyration_radius`` reduced to a Cα-only selection
(equal masses → the centroid-RMS above), reimplemented here in pure numpy to keep this
module dependency-free and consistent with the other shaping terms.

Saturating reward (anti-collapse)
---------------------------------
The raw compactness is *unbounded above* — a collapsed dense blob scores arbitrarily
high — so rewarding it directly invites a fold-collapse reward hack. Instead this term
**saturates** at a native-anchored target, exactly mirroring the linguistic-complexity
reward (:func:`._diversity_reward.lc_saturating_reward`):

    R = clip(compactness / rog_full, 0, 1)

Full credit once ``compactness >= rog_full`` (no pressure to over-compact past the
native fold state), ramping to 0 as the binder over-extends. ``rog_full`` defaults to
``0.76`` — the passing-Complexa compactness at ``r0 = 2.2`` — so the term pulls the
over-extended trained arms toward native compactness and is inert on already-globular
designs.

Like the other shaping terms it is pure numpy (no torch / Protenix / trl),
unit-testable in isolation, inert (contributes exactly 0 via its trainer weight) when
unused, and used as a **scalar** in GRPO (advantage-weighted log-prob) — it is *not*
back-propagated — so it need not be autograd differentiable, only well-behaved.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rog_compactness", "rog_compactness_reward"]

# Compact-globule Rg prefactor: Rg_compact = R0_COMPACT · N^(1/3). Calibrated so a
# typical globular protein scores ~0.7 and the passing Proteina-Complexa binder
# references score ~0.76 (Table 6 geometry analysis).
R0_COMPACT = 2.2
# Native-anchored saturation target (passing-Complexa compactness at r0 = 2.2).
ROG_FULL = 0.76


def rog_compactness(ca: np.ndarray, *, r0: float = R0_COMPACT) -> float:
    """Length-normalized compactness of a set of Cα coordinates.

    Parameters
    ----------
    ca : np.ndarray, shape ``(N, 3)``
        Cα coordinates of the chain to score.
    r0 : float
        Compact-globule Rg prefactor (``Rg_compact = r0 · N^(1/3)``).

    Returns
    -------
    float
        ``compactness = r0 · N^(1/3) / Rg_actual`` (~0.7 globular, higher = more
        compact). ``0.0`` when fewer than 2 coordinates are supplied.
    """
    ca = np.asarray(ca, dtype=np.float64)
    n = ca.shape[0]
    if n < 2:
        return 0.0
    rg_actual = float(np.sqrt(((ca - ca.mean(axis=0)) ** 2).sum(axis=-1).mean()))
    if rg_actual <= 0.0:
        return 0.0
    rg_compact = r0 * (n ** (1.0 / 3.0))
    return rg_compact / rg_actual


def rog_compactness_reward(
    coords_full: np.ndarray,
    valid_mask,
    binder_mask,
    *,
    r0: float = R0_COMPACT,
    rog_full: float = ROG_FULL,
) -> tuple[float, dict]:
    """Per-design binder radius-of-gyration compactness reward + diagnostics.

    Parameters
    ----------
    coords_full : np.ndarray, shape ``(L, 3, 3)``
        Decoded backbone in ``[N, CA, C]`` atom order for the full (padded) length.
    valid_mask, binder_mask : array-like of bool, shape ``(L,)``
        Valid-position and binder-chain masks. The compactness is computed over the
        Cα atoms at ``valid_mask & binder_mask`` (the antigen is pinned, so only the
        designed binder's fold state is scored).
    r0 : float
        Compact-globule Rg prefactor (``Rg_compact = r0 · N^(1/3)``).
    rog_full : float
        Saturation target: full credit once ``compactness >= rog_full``.

    Returns
    -------
    (term, diag) : tuple[float, dict]
        ``term = clip(compactness / rog_full, 0, 1) ∈ [0, 1]`` (higher = more
        compact, saturating at the native target). ``diag`` carries the raw
        ``compactness``, ``rg`` (``Rg_actual``, Å) and ``n_res`` (binder residues
        scored). A binder with < 2 valid residues yields ``term = 0`` and
        ``compactness = 0``.
    """
    coords_full = np.asarray(coords_full, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    binder_mask = np.asarray(binder_mask, dtype=bool)

    bpos = np.nonzero(valid_mask & binder_mask)[0]
    n = int(bpos.size)
    if n < 2:
        return 0.0, {"compactness": 0.0, "rg": 0.0, "n_res": n}

    ca = coords_full[bpos, 1, :]  # (N, 3) binder Cα (atom index 1)
    rg_actual = float(np.sqrt(((ca - ca.mean(axis=0)) ** 2).sum(axis=-1).mean()))
    compactness = rog_compactness(ca, r0=r0)

    denom = rog_full if rog_full > 0 else 1.0
    term = float(np.clip(compactness / denom, 0.0, 1.0))
    return term, {"compactness": float(compactness), "rg": rg_actual, "n_res": n}
