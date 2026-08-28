"""Tests for the GRPO within-group Jaccard novelty diversity reward.

The reward reformulation (see ``rewards/README.md`` §3) replaced the old
bonus + Foldseek cluster-rarity + two-sided anti-degeneracy hinge with a single,
uniform signal — mean pairwise k-mer-Jaccard *distance* of each design against the
rest of its group — applied independently to the amino-acid sequence and the 3Di
structural-token string. These functions are pure string math (no torch, Protenix,
or external binaries).
"""

from __future__ import annotations

import pytest

from lobster.rl_training.rewards._diversity_reward import (
    coverage,
    hamming_novelty_group,
    jaccard_novelty_group,
    kmer_jaccard,
    lc_floor_penalty,
    lc_saturating_reward,
    linguistic_complexity,
)


def test_kmer_jaccard() -> None:
    assert kmer_jaccard("ACDEFG", "ACDEFG") == 1.0  # identical
    assert kmer_jaccard("AAAA", "CCCC") == 0.0  # disjoint
    assert kmer_jaccard("", "") == 1.0  # both empty
    mid = kmer_jaccard("ACDEFG", "ACDEFH")
    assert 0.0 < mid < 1.0  # partial overlap


def test_kmer_jaccard_short_sequences() -> None:
    # A sequence shorter than k falls back to the whole-string token.
    assert kmer_jaccard("AB", "AB", k=3) == 1.0
    assert kmer_jaccard("AB", "CD", k=3) == 0.0


def test_novelty_identical_group_is_zero() -> None:
    # All identical -> every design shares all k-mers -> novelty 0 for each.
    nov = jaccard_novelty_group(["ACDEFGHIKL", "ACDEFGHIKL", "ACDEFGHIKL"])
    assert nov == [pytest.approx(0.0)] * 3


def test_novelty_disjoint_group_is_one() -> None:
    # Pairwise-disjoint k-mer sets -> maximal novelty (1.0) for each.
    nov = jaccard_novelty_group(["AAAAAA", "CCCCCC", "DDDDDD"])
    assert nov == [pytest.approx(1.0)] * 3


def test_novelty_is_mean_not_max() -> None:
    # One design identical to a peer + one disjoint peer: mean distance = 0.5,
    # which distinguishes the mean semantics from the old ``1 - max`` (would be 0).
    seqs = ["ACDEFGHIKL", "ACDEFGHIKL", "WYWYWYWYWY"]
    nov = jaccard_novelty_group(seqs)
    assert nov[0] == pytest.approx(0.5)
    assert nov[1] == pytest.approx(0.5)
    # The odd-one-out shares nothing with either peer -> novelty 1.0.
    assert nov[2] == pytest.approx(1.0)


def test_novelty_singleton_and_empty() -> None:
    assert jaccard_novelty_group(["ACDE"]) == [1.0]  # nothing to be redundant with
    assert jaccard_novelty_group([]) == []


def test_novelty_bounds() -> None:
    nov = jaccard_novelty_group(["ACDEFG", "ACDEFH", "MNPQRS", "WYWYWY"])
    assert all(0.0 <= v <= 1.0 for v in nov)


# --- pairwise Hamming novelty (between-design, positional) -----------------------------


def test_hamming_identical_group_is_zero() -> None:
    # Verbatim copies -> 0 positional difference for each.
    ham = hamming_novelty_group(["ACDEFGHIKL", "ACDEFGHIKL", "ACDEFGHIKL"])
    assert ham == [pytest.approx(0.0)] * 3


def test_hamming_all_differ_everywhere_is_one() -> None:
    # No two designs share any aligned position -> Hamming 1.0 for each.
    ham = hamming_novelty_group(["AAAA", "CCCC", "DDDD"])
    assert ham == [pytest.approx(1.0)] * 3


def test_hamming_half_positions_differ() -> None:
    # Two designs differ at exactly half the (constant-length) positions.
    ham = hamming_novelty_group(["AAAA", "AACC"])
    assert ham[0] == pytest.approx(0.5)
    assert ham[1] == pytest.approx(0.5)


def test_hamming_catches_consensus_jaccard_underreports() -> None:
    # Same vocabulary {A,C} in each design but shuffled positionally: k-mer-Jaccard sees
    # shared vocab and under-reports, Hamming (positional) sees the mismatch.
    seqs = ["ACACACAC", "CACACACA"]
    ham = hamming_novelty_group(seqs)
    assert ham[0] == pytest.approx(1.0)  # differs at every aligned position
    # Jaccard on the same pair is far lower (shared 1-mers, overlapping k-mer vocab).
    assert jaccard_novelty_group(seqs, k=1)[0] < ham[0]


def test_hamming_singleton_and_empty() -> None:
    assert hamming_novelty_group(["ACDE"]) == [1.0]
    assert hamming_novelty_group([]) == []


def test_hamming_unequal_length_uses_shorter() -> None:
    # Robust to a stray length mismatch: compare over the shorter length, no IndexError.
    ham = hamming_novelty_group(["AAAA", "AAAACCCC"])
    assert ham[0] == pytest.approx(0.0)  # AAAA matches the AAAA prefix
    assert ham[1] == pytest.approx(0.0)


# --- linguistic complexity + floor penalty (within-sequence) ---------------------------


def test_coverage_homopolymer_vs_diverse() -> None:
    # Homopolymer: 1 distinct 1-mer over cap min(20, W).
    assert coverage("AAAAAAAAAA", 1) == pytest.approx(1.0 / 10.0)
    # All-distinct residues at k=1: cap is min(20, W)=10, 10 distinct -> 1.0.
    assert coverage("ACDEFGHIKL", 1) == pytest.approx(1.0)


def test_coverage_short_sequence_is_neutral() -> None:
    # Sequence too short for the order -> neutral 1.0 (cannot signal degeneracy).
    assert coverage("AC", 3) == 1.0


def test_lc_homopolymer_is_low() -> None:
    poly = linguistic_complexity("A" * 40)
    diverse = linguistic_complexity("ACDEFGHIKLMNPQRSTVWY" * 2)
    assert poly < 0.05
    assert diverse > poly
    assert 0.0 <= poly <= 1.0 and 0.0 <= diverse <= 1.0


def test_lc_floor_penalty_bands() -> None:
    # A healthy diverse design sits above lc_hi -> zero penalty; a homopolymer saturates.
    diverse = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"  # protein-like, LC ~ 0.81
    poly = "A" * 40
    pens, lcs = lc_floor_penalty([diverse, poly], lc_hi=0.45, lc_lo=0.15)
    assert pens[0] == pytest.approx(0.0)  # above the band -> no force on healthy designs
    assert pens[1] == pytest.approx(1.0)  # full collapse -> saturated penalty
    assert lcs[0] > lcs[1]
    assert all(0.0 <= g <= 1.0 for g in pens)


def test_lc_floor_penalty_is_monotone_in_lc() -> None:
    # Penalty must be non-increasing in LC and follow the exact ramp on the band interior.
    lc_hi, lc_lo = 0.45, 0.15
    seqs = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ",  # protein-like (high LC, above band)
        "ACDEFGHIKLMNPQRSTVWY" * 2,  # composition-preserving repeat (mid LC)
        "AAAAAAAACDAAAAAAAAAA",  # near-homopolymer (low LC)
        "A" * 40,  # homopolymer (min LC)
    ]
    pens, lcs = lc_floor_penalty(seqs, lc_hi=lc_hi, lc_lo=lc_lo)
    # Non-increasing penalty as LC increases.
    order = sorted(range(len(seqs)), key=lambda i: lcs[i])
    pens_by_lc = [pens[i] for i in order]
    assert pens_by_lc == sorted(pens_by_lc, reverse=True)
    # Each penalty matches the clipped ramp on its own LC.
    for g, lc in zip(pens, lcs):
        expected = min(1.0, max(0.0, (lc_hi - lc) / (lc_hi - lc_lo)))
        assert g == pytest.approx(expected)


def test_lc_floor_penalty_degenerate_denominator() -> None:
    # lc_hi == lc_lo collapses to a hard step at the threshold (no divide-by-zero).
    pens, _ = lc_floor_penalty(["A" * 40, "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"], lc_hi=0.3, lc_lo=0.3)
    assert pens[0] == pytest.approx(1.0)  # below threshold -> full penalty
    assert pens[1] == pytest.approx(0.0)  # above threshold -> none


def test_lc_saturating_reward_saturates_and_ramps() -> None:
    # Full credit once LC >= lc_full; a homopolymer ramps toward 0. Reward is in [0, 1].
    diverse = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"  # LC ~ 0.81 (>= 0.7)
    poly = "A" * 40  # LC ~ 0.1
    rews, lcs = lc_saturating_reward([diverse, poly], lc_full=0.7)
    assert rews[0] == pytest.approx(1.0)  # already complex -> saturated, no extra pressure
    assert rews[1] < 0.25  # collapse -> small reward
    assert lcs[0] > lcs[1]
    assert all(0.0 <= r <= 1.0 for r in rews)


def test_lc_saturating_reward_is_exact_clipped_ramp() -> None:
    # r_i == clip(LC_i / lc_full, 0, 1) exactly, and non-decreasing in LC.
    lc_full = 0.6
    seqs = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ",  # high LC (saturates)
        "ACDEFGHIKLMNPQRSTVWY" * 2,  # composition-preserving repeat (mid LC)
        "AAAAAAAACDAAAAAAAAAA",  # near-homopolymer (low LC)
        "A" * 40,  # homopolymer (min LC)
    ]
    rews, lcs = lc_saturating_reward(seqs, lc_full=lc_full)
    for r, lc in zip(rews, lcs):
        assert r == pytest.approx(min(1.0, max(0.0, lc / lc_full)))
    order = sorted(range(len(seqs)), key=lambda i: lcs[i])
    rews_by_lc = [rews[i] for i in order]
    assert rews_by_lc == sorted(rews_by_lc)  # non-decreasing in LC


def test_lc_saturating_reward_degenerate_full() -> None:
    # lc_full <= 0 grants full credit to everything (no divide-by-zero).
    rews, _ = lc_saturating_reward(["A" * 40, "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ"], lc_full=0.0)
    assert rews == [pytest.approx(1.0), pytest.approx(1.0)]


def test_lc_saturating_reward_is_floor_penalty_complement() -> None:
    # The saturating reward is the positive dual of the floor penalty: on the shared ramp
    # (single-threshold at lc_full == lc_hi, lc_lo == 0), r_i == 1 - g_i.
    seqs = ["A" * 40, "AAAAAAAACDAAAAAAAAAA", "ACDEFGHIKLMNPQRSTVWY" * 2]
    thr = 0.5
    rews, _ = lc_saturating_reward(seqs, lc_full=thr)
    pens, _ = lc_floor_penalty(seqs, lc_hi=thr, lc_lo=0.0)
    for r, g in zip(rews, pens):
        assert r == pytest.approx(1.0 - g)


def test_lc_saturating_reward_on_3di_alphabet() -> None:
    # The new 3Di-token LC reward (config `w_struct_complexity`) feeds the *decoded 3Di
    # token string* through this same alphabet-generic saturating reward. 3Di is a 20-state
    # alphabet (lowercase Foldseek letters) — a monotonous D-dominated 3Di string (our arms'
    # failure mode) must score below a diverse one, exactly as for the AA track.
    di_alphabet = "acdefghiklmnpqrstvwy"  # 20 Foldseek 3Di states
    mono = "d" * 40  # D-dominated collapse
    diverse = di_alphabet * 2
    rews, lcs = lc_saturating_reward([diverse, mono], lc_full=0.7)
    assert rews[0] > rews[1]
    assert rews[1] < 0.01  # homopolymer -> LC ~0 -> ~no credit
    assert lcs[0] > lcs[1]
