"""ESMFold binder-monomer self-consistency TM-score reward (Protenix-free).

Fold the **binder sequence alone** (monomer) with ESMFold and score the TM-score between
the LeFlur-generated binder backbone (what the sampler drew) and the ESMFold-predicted
backbone (Cα, Kabsch-superposed). Higher sc-TM = the designed sequence encodes the fold
the design was drawn with. This is the *monomer* analog of the Protenix-based
``w_sctm_binder`` structure reward (:mod:`._structure_reward`), but computed by a single
ESMFold forward on the repack worker's GPU instead of a Protenix worker-pool round-trip.

Everything here is pure numpy (Kabsch + TM-score reused from :mod:`._structure_reward`) so
the policy side imports it without pulling in ``torch`` / ``transformers``. The heavy
ESMFold forward lives in the repack worker (``scripts/ligandmpnn_repack_server.py``,
``EsmFoldScorer``), exactly as the ProteinMPNN AAR forward does; this module only holds the
numpy scoring (:func:`sctm_from_ca`), the scalar reward mapper (:func:`reward_from_esmfold`),
and the sequence-degeneracy diagnostic (:func:`seq_degeneracy`).

.. warning::
   Offline this term GATED **NEGATIVE** twice (memory ``esmfold-selfconsist-gate-negative``):
   on the Complexa pool it is *anti-predictive* of pass (docking-blind), and within our own
   base pool it is univariate-predictive only as a low-sc-TM floor that is **redundant** with
   the incumbent ``rog`` + ``chainbreak`` geometry terms (clean-backbone AUROC 0.465). It is
   wired here at explicit user direction to test empirically whether the extra signal helps in
   the GRPO loop despite the offline verdict. Track the degeneracy diagnostic alongside it — a
   degenerate (single-AA-collapsed) sequence folds confidently and would inflate sc-TM.
"""

from __future__ import annotations

import math

import numpy as np

from ._structure_reward import tm_score

_STANDARD_AA1 = set("ACDEFGHIKLMNPQRSTVWY")


def sctm_from_ca(gen_ca: np.ndarray | None, pred_ca: np.ndarray | None) -> float:
    """Self-consistency TM-score between generated and ESMFold-predicted binder Cα.

    ``pred_ca`` is Kabsch-superposed onto ``gen_ca`` (both ``(L, 3)``, one-to-one residue
    correspondence) and the TM-score computed with the standard length-dependent ``d0``
    (see :func:`~plm_design_rl.rewards._structure_reward.tm_score`). A missing or
    length-mismatched pair scores ``0.0`` — so a design ESMFold could not fold simply gets
    the floor reward.

    Parameters
    ----------
    gen_ca, pred_ca : np.ndarray | None
        Binder-chain Cα coordinates ``(L, 3)`` — LeFlur-generated vs ESMFold-predicted.

    Returns
    -------
    float
        TM-score in ``[0, 1]`` (``0.0`` when inputs are missing/mismatched).
    """
    if gen_ca is None or pred_ca is None:
        return 0.0
    g = np.asarray(gen_ca, dtype=np.float64)
    p = np.asarray(pred_ca, dtype=np.float64)
    if g.ndim != 2 or g.shape != p.shape or g.shape[0] < 3 or g.shape[1] != 3:
        return 0.0
    return float(tm_score(p, g))


def reward_from_esmfold(res: dict | None) -> float:
    """Scalar sc-TM reward from a worker result dict: ``sctm`` ∈ [0, 1], or 0 if missing.

    Direct maximization — the caller multiplies this by ``w_sctm_esmfold`` and adds it as a
    flat reward term (mirroring the Protenix ``w_sctm_binder`` term), so a missing/failed
    fold drops the term to the floor rather than crashing the group.

    Parameters
    ----------
    res : dict | None
        The worker's per-design ``esmfold`` sub-dict (``{"sctm", "sc_rmsd"}``) or ``None``.

    Returns
    -------
    float
        ``clip(sctm, 0, 1)``; ``0.0`` when the design was missing/failed.
    """
    if res is None:
        return 0.0
    v = res.get("sctm")
    if v is None or not np.isfinite(v):
        return 0.0
    return float(np.clip(v, 0.0, 1.0))


def seq_degeneracy(seq: str) -> tuple[float, float]:
    """(Shannon entropy in bits, max single-AA fraction) of an amino-acid string.

    The degeneracy guard/diagnostic for the sc-TM reward: a sequence collapsed onto one
    residue type folds confidently in isolation and would inflate sc-TM without being a real
    binder (memory ``degenerate-sequences-mark-failure``). Tracking both per group separates
    "sc-TM rose because folds improved" from "sc-TM rose because sequences collapsed".

    Parameters
    ----------
    seq : str
        Binder amino-acid sequence (1-letter). Non-standard letters are ignored.

    Returns
    -------
    tuple[float, float]
        ``(entropy_bits, max_aa_frac)`` — Shannon entropy of the AA composition and the
        fraction of the most common residue. ``(nan, nan)`` for an empty/degenerate-parse
        sequence.
    """
    letters = [c for c in seq if c in _STANDARD_AA1]
    if not letters:
        return float("nan"), float("nan")
    counts: dict[str, int] = {}
    for c in letters:
        counts[c] = counts.get(c, 0) + 1
    n = len(letters)
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    return float(entropy), float(max(counts.values()) / n)
