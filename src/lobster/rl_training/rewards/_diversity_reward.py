"""Within-group k-mer-Jaccard novelty rewards for LeFlur GRPO.

The Protenix co-folding reward (:mod:`lobster.rl_training.rewards._protenix_reward`)
is a per-design docking score and says nothing about *variety* within a GRPO group.
Left unchecked, the policy reward-hacks toward a single degenerate mode — memory
records that >50%-single-AA binders make up a large slice of the budget yet pass at
~1/5 the base rate, and poly-Ala designs pass at 0%.

This module supplies a single, uniform novelty signal — mean pairwise k-mer-Jaccard
*distance* of a design against the rest of its group — applied independently to two
alphabets by the trainer:

* the **amino-acid** sequence (sequence diversity), and
* the per-residue **3Di structural-token** string (structure diversity).

A design that collapses to a degenerate mode (poly-X, or identical to its peers)
shares all its k-mers with the group and scores ~0, so a positive weight pulls the
group apart directly — no separate anti-degeneracy hinge or Foldseek clustering
(both removed; see ``README.md`` §3). All scoring is pure and operates on plain
Python strings so it is unit-testable without torch, Protenix, or external binaries.
"""


def _kmers(seq: str, k: int) -> set[str]:
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def kmer_jaccard(a: str, b: str, k: int = 3) -> float:
    """Jaccard similarity of the ``k``-mer sets of two sequences (length-robust).

    Returns 1.0 for identical (or both-empty) sequences, 0.0 for disjoint ones.
    """
    ka, kb = _kmers(a, k), _kmers(b, k)
    if not ka and not kb:
        return 1.0
    inter = len(ka & kb)
    union = len(ka | kb)
    return inter / union if union else 1.0


def jaccard_novelty_group(seqs: list[str], k: int = 3) -> list[float]:
    """Per-design novelty vs the rest of its group: ``mean_{j!=i} (1 - jaccard(i, j))``.

    Each design's novelty is the **mean** pairwise k-mer-Jaccard *distance* to every
    other design in the group. ``1`` = shares no k-mers with the group (maximally
    novel), ``0`` = identical to the group. A singleton group scores ``1.0`` (nothing
    to be redundant with).

    Used independently for the amino-acid sequence and the 3Di structural-token
    string; both are already in ``[0, 1]`` (higher = better).

    Parameters
    ----------
    seqs : list[str]
        Sequences in the GRPO group (AA strings or 3Di token strings).
    k : int, optional
        k-mer length. Defaults to 3.

    Returns
    -------
    list[float]
        One novelty score in ``[0, 1]`` per input sequence.
    """
    n = len(seqs)
    if n <= 1:
        return [1.0] * n
    out: list[float] = []
    for i in range(n):
        dist = sum(1.0 - kmer_jaccard(seqs[i], seqs[j], k) for j in range(n) if j != i)
        out.append(dist / (n - 1))
    return out
