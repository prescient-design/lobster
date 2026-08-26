"""Tests for the GRPO interface-distribution distance reward.

These cover the pure-numpy metric + histogram + reference-table machinery of
``rewards/_distribution_reward.py`` (see ``docs/leflur/grpo_distribution_reward_scope.md``).
The mini3di backbone encode (``binder_3di_states``) is exercised on GPU by the
trainer smoke, not here — the AA-only path of ``design_interface_hists`` gives full
coverage of the interface + binning + skip logic without it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lobster.rl_training.rewards._distribution_reward import (
    AA_ALPHABET,
    MIN_IFACE,
    TRIDI_ALPHABET,
    _norm,
    aa_interface_hist,
    binder_valid_flags,
    combined_distribution_terms,
    design_hists_scoped,
    design_interface_hists,
    distribution_terms,
    interface_binder_flags,
    js,
    load_reference_table,
    reference_for,
    tv,
)


# ----------------------------------------------------------------- distances
def test_norm_sums_to_one() -> None:
    p = _norm([1, 1, 2, 0])
    assert p.sum() == pytest.approx(1.0)
    assert _norm([0, 0, 0]) is None  # all-zero -> None


def test_tv_bounds_and_symmetry() -> None:
    p = _norm([1, 0, 0, 0])
    q = _norm([0, 1, 0, 0])
    assert tv(p, p) == pytest.approx(0.0)  # identical -> 0
    assert tv(p, q) == pytest.approx(1.0)  # disjoint deltas -> max
    assert tv(p, q) == pytest.approx(tv(q, p))  # symmetric
    r = _norm([1, 1, 1, 1])
    assert 0.0 <= tv(p, r) <= 1.0


def test_js_bounds_and_symmetry() -> None:
    p = _norm([1, 0, 0, 0])
    q = _norm([0, 1, 0, 0])
    assert js(p, p) == pytest.approx(0.0)  # identical -> 0
    assert js(p, q) == pytest.approx(1.0)  # disjoint -> 1 bit
    assert js(p, q) == pytest.approx(js(q, p))  # symmetric
    r = _norm([1, 1, 1, 1])
    assert 0.0 <= js(p, r) <= 1.0


# --------------------------------------------------------------- histograms
def test_aa_interface_hist_ordering_and_norm() -> None:
    # Only interface-flagged residues are tallied, in AA_ALPHABET bin order.
    seq = "ACD" + "WWW"  # last three not at interface
    flags = np.array([True, True, True, False, False, False])
    h = aa_interface_hist(seq, flags)
    assert h.sum() == pytest.approx(1.0)
    assert h[AA_ALPHABET.index("A")] == pytest.approx(1 / 3)
    assert h[AA_ALPHABET.index("C")] == pytest.approx(1 / 3)
    assert h[AA_ALPHABET.index("D")] == pytest.approx(1 / 3)
    assert h[AA_ALPHABET.index("W")] == pytest.approx(0.0)  # excluded (not interface)


def test_aa_interface_hist_empty_is_none() -> None:
    assert aa_interface_hist("ACDE", np.array([False, False, False, False])) is None


# --------------------------------------------------------------- interface
def _stack(ca: np.ndarray) -> np.ndarray:
    """(L,3) CA -> (L,3,3) [N,CA,C] with N/C offset from CA so CA stays index 1."""
    n = ca + np.array([1.0, 0.0, 0.0])
    c = ca + np.array([0.0, 1.0, 0.0])
    return np.stack([n, ca, c], axis=1)


def test_interface_flags_cross_chain_cutoff() -> None:
    # 2 antigen residues at origin; 5 binder residues, first 3 within 8 A, last 2 far.
    ca = np.array(
        [
            [0.0, 0.0, 0.0],  # antigen
            [1.0, 0.0, 0.0],  # antigen
            [2.0, 0.0, 0.0],  # binder (close)
            [3.0, 0.0, 0.0],  # binder (close)
            [4.0, 0.0, 0.0],  # binder (close)
            [50.0, 0.0, 0.0],  # binder (far)
            [60.0, 0.0, 0.0],  # binder (far)
        ]
    )
    binder = np.array([False, False, True, True, True, True, True])
    valid = np.ones(7, dtype=bool)
    flags, n_iface = interface_binder_flags(ca, binder, valid, thresh=8.0)
    # flags are aligned to binder positions in ascending order (5 binder residues).
    assert flags.tolist() == [True, True, True, False, False]
    assert n_iface == 3


def test_interface_flags_no_antigen_is_empty() -> None:
    ca = np.zeros((4, 3))
    binder = np.ones(4, dtype=bool)  # everything is binder -> no cross-chain partner
    valid = np.ones(4, dtype=bool)
    flags, n_iface = interface_binder_flags(ca, binder, valid)
    assert n_iface == 0
    assert not flags.any()


# ------------------------------------------------------ design_interface_hists
def test_design_hists_aa_only_skip_floor() -> None:
    # 3 antigen + 3 close binder => n_iface=3 < MIN_IFACE(4) -> skipped (None,None).
    ca = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=float)
    binder = np.array([False, False, False, True, True, True])
    valid = np.ones(6, dtype=bool)
    coords = _stack(ca)
    h_aa, h_3di, n_iface = design_interface_hists(coords, valid, binder, "ACD", need_aa=True, need_3di=False)
    assert n_iface < MIN_IFACE
    assert h_aa is None and h_3di is None


def test_design_hists_aa_only_passes_floor() -> None:
    # 2 antigen + 5 binder all within 8 A => n_iface=5 >= MIN_IFACE, AA hist returned.
    ca = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0]],
        dtype=float,
    )
    binder = np.array([False, False, True, True, True, True, True])
    valid = np.ones(7, dtype=bool)
    coords = _stack(ca)
    h_aa, h_3di, n_iface = design_interface_hists(coords, valid, binder, "AAACC", need_aa=True, need_3di=False)
    assert n_iface == 5
    assert h_aa is not None and h_3di is None
    assert h_aa.sum() == pytest.approx(1.0)
    assert h_aa[AA_ALPHABET.index("A")] == pytest.approx(3 / 5)
    assert h_aa[AA_ALPHABET.index("C")] == pytest.approx(2 / 5)


# --------------------------------------------------------- distribution_terms
def test_distribution_terms_identical_is_full_weight() -> None:
    ref = _norm([1, 2, 3, 4] + [0] * 16)
    term, diag = distribution_terms(None, ref.copy(), None, ref.copy(), w_aa=0.0, w_3di=0.5, metric="tv")
    assert term == pytest.approx(0.5)  # 1 - TV(0) = 1, * w_3di 0.5
    assert diag["tv_3di"] == pytest.approx(0.0)
    assert diag["js_3di"] == pytest.approx(0.0)


def test_distribution_terms_delta_vs_spread_near_zero() -> None:
    spread = _norm([1] * 20)
    delta = _norm([1] + [0] * 19)
    term, diag = distribution_terms(None, delta, None, spread, w_aa=0.0, w_3di=1.0, metric="tv")
    # TV(delta, uniform) = 1 - 1/20 = 0.95 -> term = 0.05.
    assert term == pytest.approx(0.05, abs=1e-6)
    assert diag["tv_3di"] == pytest.approx(0.95, abs=1e-6)


def test_distribution_terms_off_weight_is_zero_but_logs() -> None:
    ref = _norm([1, 1, 1, 1] + [0] * 16)
    hist = _norm([2, 1, 1, 0] + [0] * 16)
    # Weight 0 -> no reward contribution, but the raw distance is still logged.
    term, diag = distribution_terms(hist, None, ref, None, w_aa=0.0, w_3di=0.0, metric="tv")
    assert term == pytest.approx(0.0)
    assert diag["tv_aa"] is not None and diag["tv_aa"] > 0.0


def test_distribution_terms_none_hist_skipped() -> None:
    ref = _norm([1, 1, 1, 1] + [0] * 16)
    term, diag = distribution_terms(None, None, ref, ref, w_aa=1.0, w_3di=1.0, metric="tv")
    assert term == pytest.approx(0.0)
    assert diag["tv_aa"] is None and diag["tv_3di"] is None


def test_distribution_terms_js_metric() -> None:
    ref = _norm([1, 1, 1, 1] + [0] * 16)
    term, _ = distribution_terms(None, ref.copy(), None, ref.copy(), w_aa=0.0, w_3di=1.0, metric="js")
    assert term == pytest.approx(1.0)  # identical -> JS 0 -> reward 1


# ----------------------------------------------------------- reference table
def _write_ref(tmp_path, per_target, pooled, aa_alpha=AA_ALPHABET, tri_alpha=TRIDI_ALPHABET):
    p = tmp_path / "ref.json"
    p.write_text(
        json.dumps({"aa_alphabet": aa_alpha, "tridi_alphabet": tri_alpha, "per_target": per_target, "pooled": pooled})
    )
    return str(p)


def test_load_reference_and_pooled_fallback(tmp_path) -> None:
    aa = _norm([1] * 20).tolist()
    tri = _norm([2] * 20).tolist()
    pooled_aa = _norm([3] * 20).tolist()
    pooled_tri = _norm([4] * 20).tolist()
    path = _write_ref(
        tmp_path,
        {"T1": {"aa": aa, "3di": tri}},
        {"aa": pooled_aa, "3di": pooled_tri},
    )
    table = load_reference_table(path)
    # Known target -> per-target reference. Interface-only table -> binder refs None.
    r_aa, r_3di, r_aa_b, r_3di_b, src = reference_for(table, "T1")
    assert src == "per_target"
    assert np.allclose(r_aa, aa) and np.allclose(r_3di, tri)
    assert r_aa_b is None and r_3di_b is None
    # Unknown target -> pooled fallback.
    r_aa, r_3di, r_aa_b, r_3di_b, src = reference_for(table, "MISSING")
    assert src == "pooled"
    assert np.allclose(r_aa, pooled_aa) and np.allclose(r_3di, pooled_tri)
    assert r_aa_b is None and r_3di_b is None


def test_load_reference_rejects_wrong_alphabet(tmp_path) -> None:
    path = _write_ref(tmp_path, {}, {"aa": None, "3di": None}, aa_alpha="ARNDCQEGHILKMFPSTWYV")
    with pytest.raises(ValueError, match="aa_alphabet"):
        load_reference_table(path)


# ------------------------------------------------------ whole-binder scope
def test_binder_valid_flags_tracks_valid_binder_positions() -> None:
    # 2 antigen + 4 binder; last binder residue is padding (invalid).
    binder = np.array([False, False, True, True, True, True])
    valid = np.array([True, True, True, True, True, False])
    flags = binder_valid_flags(binder, valid)
    # aligned with ascending binder positions (4 of them); last is pad -> False.
    assert flags.tolist() == [True, True, True, False]


def test_design_hists_scoped_binder_superset_of_interface() -> None:
    # 2 antigen + 5 binder: first 4 close (interface), last 1 far. AA-only path.
    # (4 interface residues clears MIN_IFACE so the legacy helper also returns a hist.)
    ca = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [60, 0, 0]],
        dtype=float,
    )
    binder = np.array([False, False, True, True, True, True, True])
    valid = np.ones(7, dtype=bool)
    coords = _stack(ca)
    h_aa_i, h_3di_i, h_aa_b, h_3di_b, n_iface, n_binder = design_hists_scoped(
        coords, valid, binder, "AACDD", need_aa=True, need_3di=False
    )
    assert n_iface == 4 and n_binder == 5
    assert h_3di_i is None and h_3di_b is None  # need_3di=False -> no encode
    # Interface tallies only the 4 close residues "AACD".
    assert h_aa_i[AA_ALPHABET.index("A")] == pytest.approx(2 / 4)
    assert h_aa_i[AA_ALPHABET.index("C")] == pytest.approx(1 / 4)
    assert h_aa_i[AA_ALPHABET.index("D")] == pytest.approx(1 / 4)
    # Whole binder tallies all 5 residues "AACDD".
    assert h_aa_b[AA_ALPHABET.index("A")] == pytest.approx(2 / 5)
    assert h_aa_b[AA_ALPHABET.index("C")] == pytest.approx(1 / 5)
    assert h_aa_b[AA_ALPHABET.index("D")] == pytest.approx(2 / 5)
    # Interface subset matches the legacy interface-only helper exactly.
    h_aa_legacy, _, _ = design_interface_hists(coords, valid, binder, "AACDD", need_aa=True, need_3di=False)
    assert np.allclose(h_aa_i, h_aa_legacy)


def test_design_hists_scoped_no_min_iface_skip() -> None:
    # n_iface=3 < MIN_IFACE(4): design_interface_hists skips (None), scoped does NOT.
    ca = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=float)
    binder = np.array([False, False, False, True, True, True])
    valid = np.ones(6, dtype=bool)
    coords = _stack(ca)
    h_aa_i, _, h_aa_b, _, n_iface, n_binder = design_hists_scoped(
        coords, valid, binder, "ACD", need_aa=True, need_3di=False
    )
    assert n_iface < MIN_IFACE
    assert h_aa_i is not None and h_aa_b is not None  # unconditional, no floor
    # Legacy helper skips at the same input.
    h_legacy, _, _ = design_interface_hists(coords, valid, binder, "ACD", need_aa=True, need_3di=False)
    assert h_legacy is None


# ------------------------------------------------- combined_distribution_terms
def test_combined_alpha0_equals_interface_only() -> None:
    # α=0 must reproduce distribution_terms (interface-only) byte-for-byte.
    h_i = _norm([2, 1, 1, 0] + [0] * 16)
    h_b = _norm([1, 1, 1, 1] + [0] * 16)  # different -> would change term if α>0
    ref_i = _norm([1, 1, 1, 1] + [0] * 16)
    ref_b = _norm([4, 0, 0, 0] + [0] * 16)
    term_c, diag = combined_distribution_terms(
        None,
        h_i,
        None,
        h_b,
        None,
        ref_i,
        None,
        ref_b,
        w_aa=0.0,
        w_3di=1.0,
        alpha=0.0,
        metric="tv",
    )
    term_i, _ = distribution_terms(None, h_i, None, ref_i, w_aa=0.0, w_3di=1.0, metric="tv")
    assert term_c == pytest.approx(term_i)
    # Both scopes are still measured for logging.
    assert diag["tv_3di"] is not None and diag["tv_3di_binder"] is not None


def test_combined_alpha1_equals_binder_only() -> None:
    h_i = _norm([2, 1, 1, 0] + [0] * 16)
    h_b = _norm([1, 1, 1, 1] + [0] * 16)
    ref_i = _norm([1, 1, 1, 1] + [0] * 16)
    ref_b = _norm([1, 1, 1, 1] + [0] * 16)  # identical to h_b -> binder score 1.0
    term_c, _ = combined_distribution_terms(
        None,
        h_i,
        None,
        h_b,
        None,
        ref_i,
        None,
        ref_b,
        w_aa=0.0,
        w_3di=1.0,
        alpha=1.0,
        metric="tv",
    )
    assert term_c == pytest.approx(1.0)  # whole-binder identical -> full weight


def test_combined_alpha_half_is_mean_of_scopes() -> None:
    h_i = _norm([2, 1, 1, 0] + [0] * 16)
    h_b = _norm([3, 1, 0, 0] + [0] * 16)
    ref_i = _norm([1, 1, 1, 1] + [0] * 16)
    ref_b = _norm([1, 2, 1, 0] + [0] * 16)
    s_i = float(np.clip(1.0 - tv(h_i, ref_i), 0.0, 1.0))
    s_b = float(np.clip(1.0 - tv(h_b, ref_b), 0.0, 1.0))
    term_c, _ = combined_distribution_terms(
        None,
        h_i,
        None,
        h_b,
        None,
        ref_i,
        None,
        ref_b,
        w_aa=0.0,
        w_3di=1.0,
        alpha=0.5,
        metric="tv",
    )
    assert term_c == pytest.approx(0.5 * s_i + 0.5 * s_b)


def test_combined_missing_binder_ref_falls_back_to_interface() -> None:
    # No whole-binder reference (old interface-only table): even α=1 uses interface.
    h_i = _norm([2, 1, 1, 0] + [0] * 16)
    h_b = _norm([1, 1, 1, 1] + [0] * 16)
    ref_i = _norm([1, 1, 1, 1] + [0] * 16)
    term_c, diag = combined_distribution_terms(
        None,
        h_i,
        None,
        h_b,
        None,
        ref_i,
        None,
        None,
        w_aa=0.0,
        w_3di=1.0,
        alpha=1.0,
        metric="tv",
    )
    term_i, _ = distribution_terms(None, h_i, None, ref_i, w_aa=0.0, w_3di=1.0, metric="tv")
    assert term_c == pytest.approx(term_i)  # no dilution despite α=1
    assert diag["tv_3di_binder"] is None  # binder scope not measurable


def test_combined_zero_weight_no_reward_but_logs_both_scopes() -> None:
    h_i = _norm([2, 1, 1, 0] + [0] * 16)
    h_b = _norm([1, 1, 1, 1] + [0] * 16)
    ref_i = _norm([1, 1, 1, 1] + [0] * 16)
    ref_b = _norm([4, 0, 0, 0] + [0] * 16)
    term_c, diag = combined_distribution_terms(
        None,
        h_i,
        None,
        h_b,
        None,
        ref_i,
        None,
        ref_b,
        w_aa=0.0,
        w_3di=0.0,
        alpha=0.5,
        metric="tv",
    )
    assert term_c == pytest.approx(0.0)
    assert diag["tv_3di"] is not None and diag["tv_3di_binder"] is not None


# ------------------------------------------- reference table with binder refs
def test_load_reference_parses_binder_refs(tmp_path) -> None:
    aa = _norm([1] * 20).tolist()
    tri = _norm([2] * 20).tolist()
    aa_b = _norm([5] * 20).tolist()
    tri_b = _norm([6] * 20).tolist()
    pooled_aa_b = _norm([7] * 20).tolist()
    pooled_tri_b = _norm([8] * 20).tolist()
    path = _write_ref(
        tmp_path,
        {"T1": {"aa": aa, "3di": tri, "aa_binder": aa_b, "3di_binder": tri_b}},
        {"aa": aa, "3di": tri, "aa_binder": pooled_aa_b, "3di_binder": pooled_tri_b},
    )
    table = load_reference_table(path)
    r_aa, r_3di, r_aa_b, r_3di_b, src = reference_for(table, "T1")
    assert src == "per_target"
    assert np.allclose(r_aa_b, aa_b) and np.allclose(r_3di_b, tri_b)
    # Pooled fallback carries its own binder refs.
    _, _, r_aa_b, r_3di_b, src = reference_for(table, "MISSING")
    assert src == "pooled"
    assert np.allclose(r_aa_b, pooled_aa_b) and np.allclose(r_3di_b, pooled_tri_b)
