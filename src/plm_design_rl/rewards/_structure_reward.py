"""Structure self-consistency reward for LeFlur GRPO — self-consistency TM-score.

Measures how well the sequence the policy *designed* folds back to the backbone the
policy *generated*: TM-score between the LeFlur-decoded backbone (what the sampler
drew) and the Protenix-predicted backbone (what the oracle folds the sequence into,
already produced for the confidence reward — no extra fold call). Two versions:

* ``sctm_binder``  — binder chain only (fold self-consistency), and
* ``sctm_complex`` — the whole binder+antigen complex (fold + docking-pose
  self-consistency, since the antigen is held fixed during generation).

Everything here is pure numpy (no torch, no Protenix, no external binaries) so it is
import-light on the policy side and unit-testable without a GPU. TM-score
superposition is a vendored Kabsch alignment; see ``README.md`` §2.
"""

from __future__ import annotations

import numpy as np


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rigid-body superpose ``P`` onto ``Q`` (least-squares); return the moved ``P``.

    Both arrays are ``(L, 3)`` corresponding, equal-length ordered coordinates. The
    optimal rotation is found via SVD (with a reflection guard) after centering; the
    returned coordinates are ``P`` rotated and translated onto ``Q``'s frame.

    Parameters
    ----------
    P, Q : np.ndarray
        Shape ``(L, 3)`` coordinate sets in correspondence.

    Returns
    -------
    np.ndarray
        ``P`` superposed onto ``Q``, shape ``(L, 3)``.

    Raises
    ------
    ValueError
        If ``P`` and ``Q`` do not share shape ``(L, 3)`` with ``L >= 1``.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 1:
        raise ValueError(f"kabsch expects matching (L, 3) arrays, got {P.shape} and {Q.shape}")
    p_cen = P.mean(axis=0)
    q_cen = Q.mean(axis=0)
    Pc = P - p_cen
    Qc = Q - q_cen
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (Pc @ R.T) + q_cen


def tm_score(coords_a: np.ndarray, coords_b: np.ndarray, l_norm: int | None = None) -> float:
    """TM-score between two corresponding, equal-length ordered CA coordinate sets.

    ``coords_a`` is Kabsch-superposed onto ``coords_b`` and the TM-score computed with
    the standard length-dependent ``d0``::

        d0 = 1.24 * (L_norm - 15) ** (1/3) - 1.8   (floored at 0.5)
        TM = (1 / L_norm) * Σ_i 1 / (1 + (d_i / d0) ** 2)

    Parameters
    ----------
    coords_a, coords_b : np.ndarray
        Shape ``(L, 3)`` CA coordinates in one-to-one correspondence.
    l_norm : int | None, optional
        Length used to normalize (``d0`` + the ``1/L`` prefactor). Defaults to the
        number of aligned residues ``L``. Pass the reference length explicitly when
        normalizing a chain against a longer target.

    Returns
    -------
    float
        TM-score in ``(0, 1]``; ``0.0`` for empty or length-mismatched input.
    """
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 1 or a.shape[1] != 3:
        return 0.0
    L = a.shape[0]
    Ln = int(l_norm) if l_norm else L
    if Ln < 1:
        return 0.0
    a_sup = kabsch(a, b)
    d = np.linalg.norm(a_sup - b, axis=1)
    # d0 is undefined (imaginary cube root) for L_norm <= 15; the 0.5 floor is the
    # standard TM-align clamp for very short chains.
    if Ln > 15:
        d0 = 1.24 * (Ln - 15) ** (1.0 / 3.0) - 1.8
    else:
        d0 = 0.5
    d0 = max(d0, 0.5)
    return float(np.sum(1.0 / (1.0 + (d / d0) ** 2)) / Ln)


def structure_terms(
    gen_binder_ca: np.ndarray | None,
    pred_binder_ca: np.ndarray | None,
    gen_complex_ca: np.ndarray | None,
    pred_complex_ca: np.ndarray | None,
) -> dict[str, float]:
    """Self-consistency TM-scores for one design: binder chain and whole complex.

    Each pair is a generated (LeFlur-decoded) CA set and the corresponding
    Protenix-predicted CA set, in the same residue order. Any pair that is missing
    (``None``) or length-mismatched contributes ``0.0`` — the caller multiplies each
    by its ``w_sctm_*`` weight, so a missing structure simply drops that term.

    Parameters
    ----------
    gen_binder_ca, pred_binder_ca : np.ndarray | None
        Binder-chain CA coordinates ``(L_binder, 3)`` — generated vs predicted.
    gen_complex_ca, pred_complex_ca : np.ndarray | None
        Whole-complex CA coordinates ``(L_total, 3)`` — generated vs predicted, in a
        shared antigen-then-binder residue order.

    Returns
    -------
    dict[str, float]
        ``{"sctm_binder": float, "sctm_complex": float}`` — each TM-score in
        ``[0, 1]`` (``0.0`` when its inputs are missing/mismatched).
    """

    def _pair_tm(gen: np.ndarray | None, pred: np.ndarray | None) -> float:
        if gen is None or pred is None:
            return 0.0
        g = np.asarray(gen, dtype=np.float64)
        p = np.asarray(pred, dtype=np.float64)
        if g.ndim != 2 or g.shape != p.shape or g.shape[0] < 1:
            return 0.0
        return tm_score(g, p)

    return {
        "sctm_binder": _pair_tm(gen_binder_ca, pred_binder_ca),
        "sctm_complex": _pair_tm(gen_complex_ca, pred_complex_ca),
    }
