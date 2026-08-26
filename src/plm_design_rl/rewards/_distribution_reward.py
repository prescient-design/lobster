"""Interface-distribution distance reward for LeFlur GRPO.

Scores how close a design's **binder-interface** amino-acid and 3Di
structural-state histograms are to a per-target **reference** distribution
(Proteina-Complexa's own binders), with **no Protenix call** — a dense,
deterministic, per-step shaping signal that was validated offline to predict the
true pass objective (target-difficulty-controlled; see
``docs/leflur/grpo_distribution_reward_scope.md`` §D0).

The term slots beside the existing confidence / structure / diversity families in
:meth:`LeFlurGRPOTrainer._compute_rewards` with the identical shape — a per-design
``[0,1]`` contribution summed into ``reward_i`` — and is inert (byte-identical)
when its weights are 0.

Design → reward
---------------
For each design, over its **interface** binder residues only (binder residues with
min cross-chain Cα–Cα ``< 8 Å`` in the *generated* complex):

* ``p_aa`` — 20-bin amino-acid histogram (alphabetical ``ACDEFGHIKLMNPQRSTVWY``),
* ``p_3di`` — 20-bin mini3di structural-state histogram (Foldseek 20-state
  alphabet; the binder chain encoded **in isolation**, ``X``/unknown dropped),

then ``term = w_aa·clip(1 − D(p_aa, q_aa), 0, 1) + w_3di·clip(1 − D(p_3di,
q_3di), 0, 1)`` where ``D`` is total variation (primary) or Jensen–Shannon (bits),
and ``q`` is the per-target reference. Higher = closer to reference.

The histograms are computed by the **same** interface + binning logic as the
offline reference builder (``scripts/_tier0_compute.py``/``_cbench_compute.py``),
so a design and the reference it is scored against live in identical bin spaces.

Consistency-critical: a design whose interface collapses to one residue/state is a
delta far from any spread reference ⇒ high ``D`` ⇒ low reward, so this term
intrinsically penalizes interface degeneracy (poly-Ala/poly-Ser collapse) that the
marginal-blind k-mer diversity terms miss.

Everything except the mini3di encode is pure numpy (no torch / Protenix / trl), so
the metric math is unit-testable in isolation; the encode is a lazy import of the
in-repo Foldseek encoder, exercised on GPU by the trainer.
"""

from __future__ import annotations

import json

import numpy as np

# 20-state bin alphabets, index-aligned with the offline reference builder
# (``scripts/_tier0_compute.py``: ``AA`` and ``_iface_3di_distribution.ALPHABET``).
# AA = canonical 20 amino acids in ALPHABETICAL order (NB: this differs from the
# trainer's token-id order ``ARNDCQEGHILKMFPSTWYV`` — bin by THIS order to match
# the reference). 3Di = mini3di 20-state alphabet; the two share the same 20
# letters but index amino acids vs structural states independently.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
TRIDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
IFACE_THRESH = 8.0  # Å, cross-chain Cα–Cα (StructureComplexTransform interface def)
MIN_IFACE = 4  # designs with fewer interface residues are skipped (Tier-0 floor)

_AA_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}

# Lazily-initialized mini3di encoder (numpy-only Foldseek 3Di encoder).
_ENC = None
_CALC_CB = None


# --------------------------------------------------------------------- distances
def _norm(counts) -> np.ndarray | None:
    """Normalize a count vector to a probability histogram; ``None`` if all-zero."""
    counts = np.asarray(counts, dtype=np.float64)
    s = counts.sum()
    return counts / s if s > 0 else None


def tv(p, q) -> float:
    """Total variation distance ``½Σ|p−q| ∈ [0,1]`` (symmetric, bounded)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p - q).sum())


def js(p, q) -> float:
    """Jensen–Shannon divergence in BITS (log2) ``∈ [0,1]`` — the bounded,
    symmetric, sparse-safe (``0·log0 := 0``) stand-in for KL."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _distance(p, q, metric: str) -> float:
    return js(p, q) if metric == "js" else tv(p, q)


# ------------------------------------------------------------------- interface
def interface_binder_flags(
    ca: np.ndarray, binder_mask, valid_mask, thresh: float = IFACE_THRESH
) -> tuple[np.ndarray, int]:
    """Per-binder-residue interface boolean + interface count.

    A binder residue is at the interface iff its min Cα–Cα distance to any **valid
    antigen** residue is ``< thresh``. The returned boolean array is aligned with
    the ascending order of ``binder_mask`` positions — i.e. the exact order in
    which :meth:`LeFlurGRPOTrainer._decode_binder_seqs` / ``_decode_binder_tri``
    slice the binder — so it indexes ``seq[i]`` / ``states[i]`` directly. Binder
    positions that are not valid contribute ``False`` (never interface).

    Parameters
    ----------
    ca : np.ndarray
        ``(L, 3)`` Cα coordinates of the full (padded) complex.
    binder_mask, valid_mask : array-like of bool
        ``(L,)`` designed-binder positions and valid (non-pad) positions.
    thresh : float
        Cross-chain Cα–Cα cutoff (Å).

    Returns
    -------
    tuple[np.ndarray, int]
        ``(flags (n_binder,) bool, n_iface)``.
    """
    binder_mask = np.asarray(binder_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    bpos_all = np.nonzero(binder_mask)[0]  # binder positions in seq/state order
    apos = np.nonzero(valid_mask & ~binder_mask)[0]  # valid antigen positions
    flags = np.zeros(bpos_all.size, dtype=bool)
    if bpos_all.size == 0 or apos.size == 0:
        return flags, 0
    valid_binder_local = np.nonzero(valid_mask[bpos_all])[0]
    if valid_binder_local.size == 0:
        return flags, 0
    cb = ca[bpos_all[valid_binder_local]]  # (nvb, 3)
    cta = ca[apos]  # (na, 3)
    d = np.linalg.norm(cb[:, None, :] - cta[None, :, :], axis=-1)  # (nvb, na)
    flags[valid_binder_local] = d.min(axis=1) < thresh
    return flags, int(flags.sum())


def binder_valid_flags(binder_mask, valid_mask) -> np.ndarray:
    """Per-binder-residue *valid* boolean — the whole-binder analogue of
    :func:`interface_binder_flags`'s flag array.

    The returned boolean is aligned with the ascending order of ``binder_mask``
    positions (same order in which ``seq`` / ``states`` are sliced), and is ``True``
    at every valid (non-pad) binder residue. Passing this as the ``flags`` argument
    to :func:`aa_interface_hist` / :func:`tridi_interface_hist` tallies a histogram
    over the **entire binder chain** rather than just its interface.

    Parameters
    ----------
    binder_mask, valid_mask : array-like of bool
        ``(L,)`` designed-binder positions and valid (non-pad) positions.

    Returns
    -------
    np.ndarray
        ``(n_binder,)`` bool, ``True`` where the binder residue is valid.
    """
    binder_mask = np.asarray(binder_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    bpos_all = np.nonzero(binder_mask)[0]
    return valid_mask[bpos_all]


def aa_interface_hist(seq: str, flags: np.ndarray) -> np.ndarray | None:
    """20-bin interface amino-acid histogram (alphabetical order); ``None`` if empty."""
    counts = np.zeros(20, dtype=np.float64)
    for ch, f in zip(seq, flags):
        if f:
            j = _AA_IDX.get(ch)
            if j is not None:
                counts[j] += 1
    return _norm(counts)


# ------------------------------------------------------------------- 3Di encode
def _encoder():
    """Lazily build (and cache) the in-repo mini3di Foldseek encoder."""
    global _ENC, _CALC_CB
    if _ENC is None:
        from lobster.model.latent_generator.utils.mini3di._encoder import (
            Encoder,
            calculate_cb,
        )

        _ENC = Encoder()
        _CALC_CB = calculate_cb
    return _ENC, _CALC_CB


def binder_3di_states(coords_binder: np.ndarray) -> np.ndarray:
    """Encode a binder backbone to 3Di states.

    Parameters
    ----------
    coords_binder : np.ndarray
        ``(n_binder, 3, 3)`` backbone coords in ``[N, CA, C]`` order (the binder
        chain **in isolation** — matches the reference builder's encoding).

    Returns
    -------
    np.ndarray
        ``(n_binder,)`` int64 3Di state indices (``20`` for masked/undecodable X).
    """
    import torch

    enc, calc = _encoder()
    t = torch.from_numpy(np.ascontiguousarray(coords_binder)).float()
    Ca, Cb, N, C = calc({"coords_res": t})
    out = enc.encode_atoms(Ca, Cb, N, C)
    return out["states"].filled(20).astype(np.int64)


def tridi_interface_hist(states: np.ndarray, flags: np.ndarray) -> np.ndarray | None:
    """20-bin interface 3Di histogram (``X``/≥20 dropped); ``None`` if empty."""
    counts = np.zeros(20, dtype=np.float64)
    for s, f in zip(states, flags):
        if f and int(s) < 20:
            counts[int(s)] += 1
    return _norm(counts)


# --------------------------------------------------------------- per-design API
def design_interface_hists(
    coords_full: np.ndarray,
    valid_mask,
    binder_mask,
    seq: str,
    need_aa: bool = True,
    need_3di: bool = True,
    thresh: float = IFACE_THRESH,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Interface AA + 3Di histograms for one generated design.

    Parameters
    ----------
    coords_full : np.ndarray
        ``(L, 3, 3)`` generated backbone (full padded length) in ``[N, CA, C]``
        order (``LeFlurGRPOTrainer._decode_backbone_coords`` output for one design).
    valid_mask, binder_mask : array-like of bool
        ``(L,)`` valid (non-pad) and designed-binder positions.
    seq : str
        Binder amino-acid string, length ``== binder_mask.sum()`` and in the same
        ascending-position order (``_decode_binder_seqs`` output).
    need_aa, need_3di : bool
        Skip computing a histogram whose weight is 0 (the 3Di encode is the only
        non-trivial cost). AA is cheap and computed by default for logging.
    thresh : float
        Cross-chain interface cutoff (Å).

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None, int]
        ``(h_aa, h_3di, n_iface)``. Either histogram is ``None`` when not requested,
        the interface is empty for that alphabet, or ``n_iface < MIN_IFACE``.
    """
    coords_full = np.asarray(coords_full, dtype=np.float64)
    ca = coords_full[:, 1, :]  # CA is atom index 1
    flags, n_iface = interface_binder_flags(ca, binder_mask, valid_mask, thresh)
    if n_iface < MIN_IFACE:
        return None, None, n_iface

    h_aa = aa_interface_hist(seq, flags) if need_aa else None
    h_3di = None
    if need_3di:
        bpos_all = np.nonzero(np.asarray(binder_mask, dtype=bool))[0]
        states = binder_3di_states(coords_full[bpos_all])
        h_3di = tridi_interface_hist(states, flags)
    return h_aa, h_3di, n_iface


def design_hists_scoped(
    coords_full: np.ndarray,
    valid_mask,
    binder_mask,
    seq: str,
    need_aa: bool = True,
    need_3di: bool = True,
    thresh: float = IFACE_THRESH,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    int,
    int,
]:
    """AA + 3Di histograms for one design at **two scopes**: interface + whole-binder.

    Same machinery as :func:`design_interface_hists`, but tallies each alphabet twice —
    once over the binder *interface* (``interface_binder_flags``) and once over the
    *entire valid binder chain* (``binder_valid_flags``) — while sharing a **single**
    mini3di encode of the binder backbone across both scopes (the encode is the only
    non-trivial cost). Unlike :func:`design_interface_hists` this does **not** apply the
    ``MIN_IFACE`` skip: it returns the histograms and both counts unconditionally, and
    the caller decides how to treat a collapsed interface (the trainer's
    ``dist_min_iface`` / ``dist_iface_penalty`` guardrail).

    Parameters
    ----------
    coords_full : np.ndarray
        ``(L, 3, 3)`` generated backbone (full padded length) in ``[N, CA, C]`` order.
    valid_mask, binder_mask : array-like of bool
        ``(L,)`` valid (non-pad) and designed-binder positions.
    seq : str
        Binder amino-acid string, length ``== binder_mask.sum()``, ascending-position
        order.
    need_aa, need_3di : bool
        Skip an alphabet whose weight is 0 (the 3Di encode is the only real cost;
        AA is cheap and computed by default for logging). Skipping an alphabet nulls
        **both** its scopes.
    thresh : float
        Cross-chain interface cutoff (Å).

    Returns
    -------
    tuple
        ``(h_aa_iface, h_3di_iface, h_aa_binder, h_3di_binder, n_iface, n_binder)``.
        Any histogram is ``None`` when not requested or empty for that alphabet/scope.
    """
    coords_full = np.asarray(coords_full, dtype=np.float64)
    ca = coords_full[:, 1, :]  # CA is atom index 1
    iface_flags, n_iface = interface_binder_flags(ca, binder_mask, valid_mask, thresh)
    binder_flags = binder_valid_flags(binder_mask, valid_mask)
    n_binder = int(binder_flags.sum())

    h_aa_i = aa_interface_hist(seq, iface_flags) if need_aa else None
    h_aa_b = aa_interface_hist(seq, binder_flags) if need_aa else None
    h_3di_i = h_3di_b = None
    if need_3di:
        bpos_all = np.nonzero(np.asarray(binder_mask, dtype=bool))[0]
        states = binder_3di_states(coords_full[bpos_all])  # single shared encode
        h_3di_i = tridi_interface_hist(states, iface_flags)
        h_3di_b = tridi_interface_hist(states, binder_flags)
    return h_aa_i, h_3di_i, h_aa_b, h_3di_b, n_iface, n_binder


def distribution_terms(
    h_aa: np.ndarray | None,
    h_3di: np.ndarray | None,
    ref_aa: np.ndarray | None,
    ref_3di: np.ndarray | None,
    w_aa: float,
    w_3di: float,
    metric: str = "tv",
) -> tuple[float, dict]:
    """Weighted interface-distribution reward + raw-distance diagnostics.

    ``term = w_aa·clip(1 − D(h_aa, ref_aa), 0, 1) + w_3di·clip(1 − D(h_3di,
    ref_3di), 0, 1)`` with ``D = tv`` (default) or ``js``. A ``None`` histogram or
    reference contributes 0 (skipped). Both TV and JS are always reported (when the
    pair exists) for wandb, regardless of ``metric``.

    Returns
    -------
    tuple[float, dict]
        ``(term, {"tv_aa","tv_3di","js_aa","js_3di"})`` — diagnostics are ``None``
        where the histogram/reference pair is unavailable.
    """
    diag = {"tv_aa": None, "tv_3di": None, "js_aa": None, "js_3di": None}
    term = 0.0
    if h_aa is not None and ref_aa is not None:
        diag["tv_aa"] = tv(h_aa, ref_aa)
        diag["js_aa"] = js(h_aa, ref_aa)
        if w_aa > 0:
            term += w_aa * float(np.clip(1.0 - _distance(h_aa, ref_aa, metric), 0.0, 1.0))
    if h_3di is not None and ref_3di is not None:
        diag["tv_3di"] = tv(h_3di, ref_3di)
        diag["js_3di"] = js(h_3di, ref_3di)
        if w_3di > 0:
            term += w_3di * float(np.clip(1.0 - _distance(h_3di, ref_3di, metric), 0.0, 1.0))
    return term, diag


def combined_distribution_terms(
    h_aa_i: np.ndarray | None,
    h_3di_i: np.ndarray | None,
    h_aa_b: np.ndarray | None,
    h_3di_b: np.ndarray | None,
    ref_aa_i: np.ndarray | None,
    ref_3di_i: np.ndarray | None,
    ref_aa_b: np.ndarray | None,
    ref_3di_b: np.ndarray | None,
    w_aa: float,
    w_3di: float,
    alpha: float,
    metric: str = "tv",
) -> tuple[float, dict]:
    """Interface / whole-binder blended distribution reward + both-scope diagnostics.

    Per alphabet the closeness score ``s = clip(1 − D(hist, ref), 0, 1)`` is computed
    at the **interface** scope (``s_i``) and the **whole-binder** scope (``s_b``) and
    blended by ``alpha``::

        term = w_aa  · [ (1−α)·s_i^aa  + α·s_b^aa  ]
             + w_3di · [ (1−α)·s_i^3di + α·s_b^3di ]

    with ``α = alpha ∈ [0, 1]``. ``α = 0`` reproduces :func:`distribution_terms`
    (interface-only) byte-for-byte; ``α = 1`` is whole-binder-only; ``α = 0.5`` is
    their mean. If the **binder** histogram/reference for an alphabet is missing
    (``None`` — e.g. an old interface-only reference table), that alphabet silently
    falls back to interface-only (effective ``α = 0``) rather than diluting the reward;
    symmetrically a missing interface score falls back to binder-only. An alphabet
    whose weight is ``≤ 0`` contributes nothing to ``term`` but is still measured for
    logging.

    Both TV and JS are always reported for every available (hist, ref) pair, at both
    scopes, regardless of ``metric``.

    Returns
    -------
    tuple[float, dict]
        ``(term, diag)`` where ``diag`` has interface keys
        ``tv_aa,tv_3di,js_aa,js_3di`` and whole-binder keys
        ``tv_aa_binder,tv_3di_binder,js_aa_binder,js_3di_binder`` (``None`` where the
        corresponding pair is unavailable).
    """
    diag = {
        "tv_aa": None,
        "tv_3di": None,
        "js_aa": None,
        "js_3di": None,
        "tv_aa_binder": None,
        "tv_3di_binder": None,
        "js_aa_binder": None,
        "js_3di_binder": None,
    }
    term = 0.0

    def _score(h, ref):
        if h is None or ref is None:
            return None
        return float(np.clip(1.0 - _distance(h, ref, metric), 0.0, 1.0))

    def _blend(h_i, ref_i, h_b, ref_b, w, k_tv, k_js, k_tvb, k_jsb):
        nonlocal term
        if h_i is not None and ref_i is not None:
            diag[k_tv] = tv(h_i, ref_i)
            diag[k_js] = js(h_i, ref_i)
        if h_b is not None and ref_b is not None:
            diag[k_tvb] = tv(h_b, ref_b)
            diag[k_jsb] = js(h_b, ref_b)
        if w <= 0:
            return
        s_i = _score(h_i, ref_i)
        s_b = _score(h_b, ref_b)
        if s_i is None and s_b is None:
            return
        if s_b is None:  # no whole-binder ref -> interface-only (no dilution)
            blended = s_i
        elif s_i is None:  # no interface ref -> whole-binder-only
            blended = s_b
        else:
            blended = (1.0 - alpha) * s_i + alpha * s_b
        term += w * float(blended)

    _blend(h_aa_i, ref_aa_i, h_aa_b, ref_aa_b, w_aa, "tv_aa", "js_aa", "tv_aa_binder", "js_aa_binder")
    _blend(h_3di_i, ref_3di_i, h_3di_b, ref_3di_b, w_3di, "tv_3di", "js_3di", "tv_3di_binder", "js_3di_binder")
    return term, diag


# ------------------------------------------------------------- reference table
def load_reference_table(path: str) -> dict:
    """Load a per-target interface reference table written by the offline builder.

    Expected JSON schema (the ``*_binder`` keys are **optional** — a whole-binder
    reference; absent in interface-only tables and then read as ``None``)::

        {"aa_alphabet": "ACDEFGHIKLMNPQRSTVWY",
         "tridi_alphabet": "ACDEFGHIKLMNPQRSTVWY",
         "per_target": {"<target_id>": {"aa": [20]|null, "3di": [20]|null,
                                        "aa_binder": [20]|null, "3di_binder": [20]|null}, ...},
         "pooled": {"aa": [20]|null, "3di": [20]|null,
                    "aa_binder": [20]|null, "3di_binder": [20]|null}}

    Raises
    ------
    ValueError
        If a stored alphabet does not match this module's bin ordering (would make
        the design and reference histograms index-incompatible).
    """
    with open(path) as fh:
        raw = json.load(fh)
    aa_alpha = raw.get("aa_alphabet", AA_ALPHABET)
    tri_alpha = raw.get("tridi_alphabet", TRIDI_ALPHABET)
    if aa_alpha != AA_ALPHABET:
        raise ValueError(f"reference aa_alphabet {aa_alpha!r} != expected {AA_ALPHABET!r}")
    if tri_alpha != TRIDI_ALPHABET:
        raise ValueError(f"reference tridi_alphabet {tri_alpha!r} != expected {TRIDI_ALPHABET!r}")

    def _vec(v):
        return np.asarray(v, dtype=np.float64) if v is not None else None

    def _entry(v):
        # (aa_iface, 3di_iface, aa_binder, 3di_binder); *_binder default None.
        return (
            _vec(v.get("aa")),
            _vec(v.get("3di")),
            _vec(v.get("aa_binder")),
            _vec(v.get("3di_binder")),
        )

    per_target = {tid: _entry(v) for tid, v in raw.get("per_target", {}).items()}
    return {
        "per_target": per_target,
        "pooled": _entry(raw.get("pooled", {})),
        "aa_alphabet": aa_alpha,
        "tridi_alphabet": tri_alpha,
    }


def reference_for(
    table: dict, target_id: str
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """Per-target reference if present, else pooled.

    Returns
    -------
    tuple
        ``(ref_aa_iface, ref_3di_iface, ref_aa_binder, ref_3di_binder, source)``. The
        ``*_binder`` refs are ``None`` for interface-only tables (older JSONs), in which
        case the whole-binder distribution scope falls back to interface-only.
    """
    pt = table["per_target"].get(target_id)
    if pt is not None:
        return pt[0], pt[1], pt[2], pt[3], "per_target"
    aa_i, tri_i, aa_b, tri_b = table["pooled"]
    return aa_i, tri_i, aa_b, tri_b, "pooled"
