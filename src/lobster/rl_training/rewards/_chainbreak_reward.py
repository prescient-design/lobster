"""Backbone chain-break (peptide-bond integrity) shaping reward for LeFlur GRPO.

Every other structural reward — steric clash (:mod:`._clash_reward`), interface
3Di/AA distribution (:mod:`._distribution_reward`), 3DZD shape complementarity
(:mod:`._shape_reward`) — is computed on the **model-generated backbone**. But the
codec-decoded backbone can be *severed*: consecutive residues whose peptide bond
``C(i)–N(i+1)`` is stretched from its ideal ``~1.33 Å`` to tens of Å (chain
crossing / codec blow-up). On such a backbone the geometry those other rewards
measure is meaningless. Offline, ~63% of our rollout designs carry ≥1 hard break
(``C–N > 2.0 Å``) vs ~0.5% of native passing Complexa references.

This module scores a single **smooth** ``[0, 1]`` term that rewards an intact
backbone (higher = better), so the structure track emits physical geometry and the
*other* structural rewards become trustworthy. It is a **realism regularizer, not a
pass predictor** — offline AUROC vs Protenix pass ≈ 0.44, as expected: Protenix
re-folds the generated *sequence* with the antigen MSA and discards the generated
coordinates, so chain-break in the generated backbone is invisible to it. Its value
is upstream: it keeps the coordinates the policy is optimised on physically valid.

It mirrors :mod:`._clash_reward`: pure numpy (no torch / Protenix / trl),
unit-testable in isolation, inert (contributes exactly 0 via its trainer weight)
when unused, and used as a **scalar** in GRPO (advantage-weighted log-prob) — it is
*not* back-propagated — so it need not be autograd differentiable, only numerically
smooth / well-behaved.

Design → reward
---------------
For one design with decoded backbone ``coords_full (L, 3, 3)`` in ``[N, CA, C]``
order plus ``valid_mask`` / ``binder_mask`` ``(L,)``, walk the **binder** residues
in index order and score each peptide bond between *adjacent* valid binder residues
(``bpos[k+1] == bpos[k] + 1`` — the dense-array analog of same-chain resSeq
contiguity):

    d          = ‖C(i) − N(i+1)‖
    excess     = min(cap, max(0, |d − ideal| − tol))     # deadband + saturation
    r_bond(d)  = exp(−(excess / sigma)²)                 # ∈ (0, 1], 1 = ideal

* ``tol`` is a free deadband around the ideal bond length (reference bond error
  ~0.028 Å sits comfortably inside), so ordinary bond-length jitter costs nothing.
* ``cap`` **saturates** the deviation: a 70 Å break scores the same as a 3.3 Å one.
  This is the stability lever — it bounds the per-bond penalty so one catastrophic
  break cannot produce an unbounded advantage.

**Design scalars** (all ``∈ [0, 1]``, higher = better):

    mean_r     = mean_bond r_bond            # dense bulk-realism (per-token natural)
    gate       = exp(−n_break / gate_k)      # catastrophic-break gate
    R          = mean_r · gate               # the reward term ("R_meanxgate")

``n_break`` is either a **discrete** count (``gate_mode="count"``:
``Σ 1[d > break_hard]``) or a **smooth** severity-aware count
(``gate_mode="soft"``: ``Σ sigmoid((d − break_d0) / break_soft)``). The two factors
decouple concerns: ``mean_r`` rewards dense bond-length realism across all bonds
(the signal a per-token structure advantage needs), while ``gate`` collapses the
reward when any bond is catastrophically severed. The ``soft`` mode additionally
removes the reward cliff as a bond crosses the hard threshold; whether its bulk
floor is negligible enough to prefer over ``count`` is decided by the offline gate.

Per-token credit
----------------
With ``return_eres`` the module returns ``diag["cb_break_res"]`` — a per-binder-residue
penalty (each bond's ``1 − r_bond`` split 50/50 between its two endpoint residues)
whose sum equals ``pen = Σ (1 − r_bond)`` exactly, so ``mean_r = 1 − pen/n_bonds``.
The per-token structure advantage negates and normalises this (like clash's
``E_clash_res``) to route per-residue break credit into the structure track. The
gate is a design-level transform kept only in the scalar term (mirroring how clash's
``exp`` lives only in the scalar), so the per-residue signal stays the raw bond
realism — catastrophic-break residues already dominate it (``1 − r_bond ≈ 1`` vs a
bulk ``~0.07``).

Notes
-----
Scored on binder residues only (the designed chain); the antigen backbone is native
and intact. Decoded backbones are codec reconstructions (~1–2.7 Å RMSD), so the
default ``tol = 0.10`` absorbs reconstruction jitter while ``break_hard = 2.0``
flags genuine severances.
"""

from __future__ import annotations

import numpy as np

# Defaults (Å where a distance; see module docstring for the reward shapes).
CN_IDEAL = 1.33  # ideal peptide-bond C(i)-N(i+1) length
TOL = 0.10  # deadband: |d - ideal| below this costs nothing (ref err ~0.028)
CAP = 2.00  # saturate deviation: a 70 A break == a 3.3 A break (stability lever)
SIGMA = 0.50  # width of the soft-core bond well; r_bond ~0 by the 2.0 A break
GATE_K = 2.0  # break count that drops the gate to 1/e (exp(-n_break/GATE_K))
BREAK_HARD = 2.0  # count mode: C-N above this is a hard break (d > break_hard)
BREAK_D0 = 2.0  # soft mode: sigmoid center for the smooth break count
BREAK_SOFT = 0.10  # soft mode: sigmoid width (Å) for the smooth break count


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic sigmoid via the ``tanh`` identity.

    ``sigmoid(x) = 0.5·(1 + tanh(x/2))`` — evaluated with ``tanh`` so it never
    overflows for large ``|x|`` (unlike ``exp(x)`` in a two-branch ``np.where``,
    where the discarded branch still overflows on severed bonds with ``d ≫ d0``).
    """
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _r_bond(d: np.ndarray, *, ideal: float, tol: float, cap: float, sigma: float) -> np.ndarray:
    """Per-bond realism ``exp(−(excess/sigma)²) ∈ (0, 1]`` with deadband + saturation.

    Parameters
    ----------
    d : np.ndarray
        Peptide-bond ``C(i)–N(i+1)`` distances (Å).
    ideal, tol, cap, sigma : float
        Ideal bond length, free deadband, deviation saturation cap, and well width.

    Returns
    -------
    np.ndarray
        Per-bond realism in ``(0, 1]`` (1 at/inside the deadband, → its floor at the
        cap: ``exp(−(cap/sigma)²)``).
    """
    excess = np.minimum(cap, np.maximum(0.0, np.abs(d - ideal) - tol))
    return np.exp(-((excess / sigma) ** 2))


def chainbreak_reward(
    coords_full: np.ndarray,
    valid_mask,
    binder_mask,
    *,
    ideal: float = CN_IDEAL,
    tol: float = TOL,
    cap: float = CAP,
    sigma: float = SIGMA,
    gate_k: float = GATE_K,
    gate_mode: str = "count",
    break_hard: float = BREAK_HARD,
    break_d0: float = BREAK_D0,
    break_soft: float = BREAK_SOFT,
    return_eres: bool = False,
) -> tuple[float, dict]:
    """Backbone chain-break (peptide-bond integrity) reward for one generated design.

    Parameters
    ----------
    coords_full : np.ndarray
        ``(L, 3, 3)`` generated backbone (full padded length) in ``[N, CA, C]``
        order (``LeFlurGRPOTrainer._decode_backbone_coords`` output for one design).
    valid_mask, binder_mask : array-like of bool
        ``(L,)`` valid (non-pad) and designed-binder positions.
    ideal, tol, cap, sigma : float
        Per-bond realism shape (see :func:`_r_bond`): ideal ``C–N`` length,
        deadband, deviation saturation cap, and soft-core well width.
    gate_k : float
        Break count that drops the multiplicative gate to ``1/e``
        (``gate = exp(−n_break / gate_k)``).
    gate_mode : {"count", "soft"}
        ``"count"``: ``n_break = Σ 1[d > break_hard]`` (discrete hard-break count).
        ``"soft"``: ``n_break = Σ sigmoid((d − break_d0) / break_soft)`` (smooth,
        severity-aware, no reward cliff at the threshold).
    break_hard : float
        ``count`` mode hard-break threshold (Å).
    break_d0, break_soft : float
        ``soft`` mode sigmoid center (Å) and width (Å).
    return_eres : bool
        When ``True`` add ``diag["cb_break_res"]`` — the per-binder-residue break
        penalty of shape ``(n_valid_binder,)`` (in the order of the valid binder
        positions, ``np.nonzero(valid_mask & binder_mask)``), whose sum equals
        ``pen = Σ (1 − r_bond)`` **exactly**. Each peptide bond's penalty is split
        50/50 between its two endpoint residues. Used by the per-token chain-break
        advantage to assign per-residue credit to the structure track. Default
        ``False`` (byte-identical: no extra key, scalars unchanged).

    Returns
    -------
    tuple[float, dict]
        ``(term, diag)`` where ``term = mean_r · gate ∈ [0, 1]`` (unweighted,
        "R_meanxgate") and ``diag = {"mean_r", "gate", "n_break", "n_hardbreak",
        "pen", "n_bonds", "max_cn"}`` (plus ``"cb_break_res"`` when ``return_eres``).

    Notes
    -----
    A design with fewer than two adjacent valid binder residues has no peptide bond
    to score: ``mean_r = 1.0``, ``gate = 1.0`` ⇒ ``term = 1.0`` (nothing to
    penalize), ``pen = 0``, and (if requested) ``cb_break_res`` is all zeros. All
    components are continuous in the coordinates (``soft`` gate fully so; ``count``
    gate steps at the threshold, by design).

    Examples
    --------
    >>> import numpy as np
    >>> # 4 residues in a row, ideal ~1.33 A C-N spacing along x within a chain
    >>> # (N, CA, C per residue); build an intact 2-residue binder + 2 antigen res.
    >>> def res(x0):
    ...     return np.array([[x0, 0, 0], [x0 + 1.46, 0, 0], [x0 + 2.5, 0, 0]])
    >>> coords = np.stack([res(0.0), res(3.83), res(20.0), res(23.83)]).astype(float)
    >>> valid = np.ones(4, dtype=bool)
    >>> binder = np.array([False, False, True, True])
    >>> term, diag = chainbreak_reward(coords, valid, binder)
    >>> 0.0 <= term <= 1.0
    True
    """
    coords_full = np.asarray(coords_full, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    binder_mask = np.asarray(binder_mask, dtype=bool)

    bpos = np.nonzero(valid_mask & binder_mask)[0]
    e_res = np.zeros(bpos.size, dtype=np.float64) if return_eres else None

    # Peptide bonds only between residues adjacent in the dense array (== same-chain,
    # contiguous resSeq): local endpoints (k, k+1) with bpos[k+1] == bpos[k] + 1.
    mean_r = 1.0
    gate = 1.0
    n_break = 0.0
    n_hardbreak = 0
    pen = 0.0
    n_bonds = 0
    max_cn = 0.0
    if bpos.size > 1:
        adj = bpos[1:] == bpos[:-1] + 1  # (nb-1,) which consecutive pairs are bonded
        k = np.nonzero(adj)[0]  # local endpoint indices of bonded pairs
        if k.size:
            c_i = coords_full[bpos[k], 2, :]  # C of residue i (atom index 2)
            n_j = coords_full[bpos[k + 1], 0, :]  # N of residue i+1 (atom index 0)
            d = np.linalg.norm(c_i - n_j, axis=-1)  # (n_bonds,)
            rb = _r_bond(d, ideal=ideal, tol=tol, cap=cap, sigma=sigma)
            pb = 1.0 - rb
            n_bonds = int(d.size)
            mean_r = float(rb.mean())
            pen = float(pb.sum())
            max_cn = float(d.max())
            n_hardbreak = int((d > break_hard).sum())
            if gate_mode == "soft":
                n_break = float(_sigmoid((d - break_d0) / break_soft).sum())
            elif gate_mode == "count":
                n_break = float(n_hardbreak)
            else:
                raise ValueError(f"gate_mode must be 'count' or 'soft', got {gate_mode!r}")
            gate = float(np.exp(-n_break / gate_k))
            if return_eres:
                # Split each bond penalty 50/50 between its two endpoint residues.
                np.add.at(e_res, k, 0.5 * pb)
                np.add.at(e_res, k + 1, 0.5 * pb)

    term = float(mean_r * gate)
    diag = {
        "mean_r": mean_r,
        "gate": gate,
        "n_break": n_break,
        "n_hardbreak": n_hardbreak,
        "pen": pen,
        "n_bonds": n_bonds,
        "max_cn": max_cn,
    }
    if return_eres:
        diag["cb_break_res"] = e_res
    return term, diag
