"""LigandMPNN amino-acid-recovery (AAR) shaping reward for LeFlur GRPO (whole-binder).

A structure-conditioned inverse-folding model (ProteinMPNN ``v_48_020``) is asked, on the
design's *own fixed backbone* with the antigen pinned as fixed context and the binder chain
designable, how closely the design's sequence matches what it would itself place there:

* **AAR** (design mode) — ProteinMPNN redesigns the binder ``k`` times at low temperature;
  ``aar`` = mean over binder residues of ``1[sampled AA == our AA]``, averaged over draws.
  "How designable / self-consistent is our sequence given this backbone."
* **C_mpnn** (consistency) — teacher-forced ``exp(mean_i log P(bd_seq_i | backbone))``.
  Carried alongside AAR as a co-diagnostic, never the primary reward here.

Scope — the reward is over the **whole binder**
-----------------------------------------------
The scalar reward is the AAR over **all** binder residues (:func:`reward_from_aar`), per
the design directive *track the interface, but reward over the binder*. The interface-only
values (``aar_iface`` / ``c_mpnn_iface``, binder Cα within :data:`IFACE_D0` Å of any antigen
Cα) and both ``c_mpnn`` scopes are computed and returned **only as diagnostics** — they
never enter the reward.

Grounding — provided opt-in, correlation is weak (documented, not gated)
-----------------------------------------------------------------------
Unlike the steric-clash term, whole-binder AAR is *not* a hard physical constraint; it is a
designability / seq↔struct-consistency proxy. The offline discrimination study found
whole-binder AAR (and C_mpnn) to be **anti-predictive** of the Protenix binder-pass label
(AUROC ≈ 0.29–0.34): high MPNN agreement tends to mark generic / low-complexity sequences,
which fail. Interface-scope consistency was less anti-predictive. This term is therefore
kept **opt-in with default weight 0** and this caveat is documented rather than used to gate
the term out — the user asked for the AAR signal to be available; enabling its weight is a
deliberate, calibrated choice (see ``scripts/_aar_analyze.py``), not a default.

Design → reward
---------------
The heavy ProteinMPNN forward (torch) runs in the CPU repack/scoring worker, which produces
per-binder-residue arrays and reduces them with the pure-numpy :func:`aar_terms`; the policy
side maps the resulting dict to a scalar with :func:`reward_from_aar`::

    reward = aar   ∈ [0, 1]   (whole-binder recovery; 1 = every binder residue recovered)

Like the other reward terms this module is pure numpy (no torch / trl): the model forward is
kept entirely in the worker, and this module only reduces / maps the per-residue results, so
importing it stays cheap and it is directly unit-testable.
"""

from __future__ import annotations

import numpy as np

IFACE_D0 = 8.0  # binder Cα within this of any antigen Cα => "interface" residue (diagnostic)

# --- LigandMPNN/ProteinMPNN 1-letter -> LeFlur 33-token (AA_VOCAB) id remap ---------------
# The repack worker emits the LigandMPNN-designed binder sequence as a 1-letter string (the
# canonical MPNN alphabet "ACDEFGHIKLMNPQRSTVWY"). The CHORD SFT-distillation term supervises
# the policy's *sequence-track* logits, which live over the 33-token amino-acid vocabulary
# (``AA_VOCAB``), so the designed letters must be remapped into that space. Any letter with no
# canonical 20-AA slot (e.g. the MPNN "X"/unknown) maps to :data:`SFT_IGNORE_INDEX` so the CE
# skips it — we never distil an unknown/degenerate identity.
SFT_IGNORE_INDEX = -100  # torch cross-entropy ignore_index; also our "no supervision" sentinel

# Canonical 20-AA 1-letter -> AA_VOCAB (33-token) id. Built from AA_VOCAB so the ids stay in
# lock-step with the tokenizer (LeFlur's ``sequence_tokens`` are AA_VOCAB ids).
_STANDARD_AA1 = "ACDEFGHIKLMNPQRSTVWY"


def _aa1_to_aa33() -> dict[str, int]:
    """1-letter standard AA -> AA_VOCAB (33-token) id (imported lazily to keep this module light)."""
    from lobster.tokenization._amino_acid import AA_VOCAB

    return {c: int(AA_VOCAB[c]) for c in _STANDARD_AA1}


def binder_letters_to_aa33(seq_design: str) -> np.ndarray:
    """Map a designed binder 1-letter string to an ``(Nb,)`` array of AA_VOCAB ids.

    Non-standard letters (e.g. MPNN ``X``) become :data:`SFT_IGNORE_INDEX`.

    Parameters
    ----------
    seq_design : str
        LigandMPNN-designed binder sequence (1-letter, canonical AA alphabet).

    Returns
    -------
    np.ndarray
        ``(len(seq_design),)`` int64 array of AA_VOCAB token ids, ``SFT_IGNORE_INDEX``
        where the letter has no standard-AA slot.
    """
    m = _aa1_to_aa33()
    return np.array([m.get(c, SFT_IGNORE_INDEX) for c in seq_design], dtype=np.int64)


def interface_residue_mask(bd_ca: np.ndarray, ag_ca: np.ndarray, d0: float = IFACE_D0) -> np.ndarray:
    """Boolean ``(Lb,)`` mask of binder residues at the interface.

    A binder residue is "at the interface" if its Cα lies within ``d0`` Å of any antigen Cα
    (matching the offline ``scripts/_aar_compute.py`` convention). Numpy mirror of the
    worker's torch ``cdist`` so the reduction is testable without a GPU.

    Parameters
    ----------
    bd_ca : np.ndarray
        ``(Lb, 3)`` binder Cα coordinates.
    ag_ca : np.ndarray
        ``(La, 3)`` antigen Cα coordinates.
    d0 : float
        Cα–Cα interface cutoff (Å).

    Returns
    -------
    np.ndarray
        ``(Lb,)`` boolean mask (all ``False`` if either chain is empty).
    """
    lb = int(bd_ca.shape[0])
    if lb == 0 or ag_ca.shape[0] == 0:
        return np.zeros(lb, dtype=bool)
    d = np.sqrt(np.maximum(((bd_ca[:, None, :] - ag_ca[None, :, :]) ** 2).sum(-1), 1e-12))
    return d.min(axis=1) < d0


def aar_terms(
    match_res: np.ndarray,
    logp_res: np.ndarray,
    binder_mask: np.ndarray,
    iface_mask: np.ndarray | None = None,
) -> dict:
    """Reduce per-residue ProteinMPNN outputs to whole-binder + interface AAR / consistency.

    Pure-numpy reduction the scoring worker calls after ProteinMPNN produces the per-residue
    recovery and teacher-forced log-probabilities. Mirrors the offline
    ``scripts/_aar_compute.py`` reductions exactly (whole binder = mean over
    ``binder_mask``; interface = mean over ``iface_mask``; ``c_mpnn = exp(mean logp)``).

    Parameters
    ----------
    match_res : np.ndarray
        ``(L,)`` per-residue recovery in ``[0, 1]`` (``1[sampled == ours]`` averaged over
        the ``k`` design draws).
    logp_res : np.ndarray
        ``(L,)`` per-residue teacher-forced ``log P(our AA | backbone)`` (averaged over
        draws). Used for the C_mpnn diagnostic only.
    binder_mask : np.ndarray
        ``(L,)`` boolean/0-1 mask selecting valid binder residues (designable & valid).
    iface_mask : np.ndarray | None
        ``(L,)`` boolean/0-1 mask selecting interface binder residues (subset of
        ``binder_mask``). ``None`` or all-``False`` → interface diagnostics are ``nan``.

    Returns
    -------
    dict
        Reward input ``aar`` (whole-binder recovery) plus the mapped scalar ``term``, and
        diagnostics ``aar_iface`` / ``c_mpnn`` / ``c_mpnn_iface`` / ``nll`` / ``logp_mean``
        / ``n_binder`` / ``n_iface``. Whole-binder metrics are ``nan`` when there are no
        binder residues (the mapper floors ``nan`` to 0.0).
    """
    match_res = np.asarray(match_res, dtype=np.float64).reshape(-1)
    logp_res = np.asarray(logp_res, dtype=np.float64).reshape(-1)
    bm = np.asarray(binder_mask).astype(bool).reshape(-1)
    im = np.zeros_like(bm) if iface_mask is None else np.asarray(iface_mask).astype(bool).reshape(-1)
    im = im & bm  # interface is always a subset of the binder

    n_binder = int(bm.sum())
    n_iface = int(im.sum())
    nan = float("nan")
    out = {
        "aar": nan,
        "aar_iface": nan,
        "c_mpnn": nan,
        "c_mpnn_iface": nan,
        "nll": nan,
        "logp_mean": nan,
        "n_binder": n_binder,
        "n_iface": n_iface,
        "term": 0.0,
    }
    if n_binder == 0:
        return out

    out["aar"] = float(match_res[bm].mean())
    mean_logp = float(logp_res[bm].mean())
    out["c_mpnn"] = float(np.exp(mean_logp))
    out["nll"] = -mean_logp
    out["logp_mean"] = mean_logp
    if n_iface > 0:
        out["aar_iface"] = float(match_res[im].mean())
        out["c_mpnn_iface"] = float(np.exp(float(logp_res[im].mean())))
    out["term"] = reward_from_aar(out)
    return out


def reward_from_aar(res: dict | None) -> float:
    """Scalar AAR reward = whole-binder recovery ``aar`` ∈ [0, 1].

    Bounded to ``[0, 1]`` and never negative. A missing/failed design (``None``), a missing
    ``aar`` key, or a ``nan`` ``aar`` (no binder residues) floors to ``0.0``.

    Parameters
    ----------
    res : dict | None
        Metrics dict from :func:`aar_terms` (uses ``aar``). ``None`` → 0.0.

    Returns
    -------
    float
        Reward in ``[0, 1]``.
    """
    if res is None:
        return 0.0
    aar = res.get("aar")
    if aar is None or not np.isfinite(aar):
        return 0.0
    return float(min(1.0, max(0.0, float(aar))))
