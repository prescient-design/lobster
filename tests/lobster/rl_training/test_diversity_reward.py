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
    jaccard_novelty_group,
    kmer_jaccard,
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
