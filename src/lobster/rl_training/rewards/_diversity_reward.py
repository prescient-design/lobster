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


def hamming_novelty_group(seqs: list[str]) -> list[float]:
    """Per-design mean normalized Hamming distance to the rest of its group.

    A GRPO group targets one epitope at one fixed binder length, so the group is a set
    of equal-length strings and the position-wise Hamming distance is well-defined:

        h_i = mean_{j != i} (1 / L) * sum_l 1[a_{i,l} != a_{j,l}]   in [0, 1].

    ``h_i = 0`` when design ``i`` is a verbatim copy of every peer, ``h_i = 1`` when it
    differs at every aligned position. Unlike the k-mer-Jaccard novelty (which compares
    k-mer *vocabularies*, order-blind), Hamming is *positional* — it fires on consensus
    convergence, the group agreeing residue-by-residue toward one sequence, which a
    shared-vocabulary Jaccard can under-report. A singleton group scores ``1.0``.

    Lengths are equal within a real GRPO group; if two designs differ in length (e.g. a
    truncated decode) the shorter length is used for that pair, counting only aligned
    positions — never an index error.

    Parameters
    ----------
    seqs : list[str]
        Sequences in the GRPO group (AA strings or 3Di token strings), nominally equal
        length.

    Returns
    -------
    list[float]
        One Hamming-novelty score in ``[0, 1]`` per input sequence.
    """
    n = len(seqs)
    if n <= 1:
        return [1.0] * n
    out: list[float] = []
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if j == i:
                continue
            a, b = seqs[i], seqs[j]
            m = min(len(a), len(b))
            if m == 0:
                acc += 1.0 if (a or b) else 0.0
                continue
            acc += sum(1 for x, y in zip(a, b) if x != y) / m
        out.append(acc / (n - 1))
    return out


def coverage(seq: str, k: int) -> float:
    """Order-``k`` coverage: distinct k-mers / min(20^k, W), W = len(seq) - k + 1.

    Returns ``1.0`` (neutral) when the sequence is too short for order ``k`` (``W < 1``):
    an unreachable order cannot signal degeneracy. The linguistic-complexity factor for
    order ``k`` (exposed for per-order ``cov^(k)`` tracking).
    """
    w = len(seq) - k + 1
    if w < 1:
        return 1.0
    u = len({seq[i : i + k] for i in range(w)})
    cap = min(20**k, w)
    return u / cap if cap > 0 else 1.0


def linguistic_complexity(seq: str, kmax: int = 3) -> float:
    """Per-design linguistic complexity ``LC = prod_{k=1..kmax} cov^(k)`` in ``[0, 1]``.

    The product is a conjunction across orders: a deficit at *any* ``k`` multiplies the
    score down, so ``LC`` is small for single-residue collapse (kills ``cov^(1)``),
    distributed few-residue mush (kills ``cov^(1),cov^(2)``), and composition-preserving
    repeats (kills ``cov^(2),cov^(3)`` while ``cov^(1)`` stays high) alike. Truncating at
    ``kmax=3`` drops the saturated high-order factors (``~1`` for every design) that only
    add noise. Higher = more complex; healthy references sit near ``0.52``--``0.57``,
    poly-alanine collapse near ``0.10``.

    Parameters
    ----------
    seq : str
        A single decoded binder sequence.
    kmax : int, optional
        Highest k-mer order in the product. Defaults to 3.

    Returns
    -------
    float
        Linguistic complexity in ``[0, 1]``.
    """
    lc = 1.0
    for k in range(1, kmax + 1):
        lc *= coverage(seq, k)
    return lc


def lc_floor_penalty(
    seqs: list[str], lc_hi: float = 0.45, lc_lo: float = 0.15, kmax: int = 3
) -> tuple[list[float], list[float]]:
    """Per-design linguistic-complexity floor penalty ``g_i`` and the raw ``LC_i``.

    ``LC_i`` is turned into a one-sided soft hinge that is *identically zero inside the
    healthy band* and ramps to ``1`` at full collapse — a two-threshold linear ramp:

        g_i = clip((lc_hi - LC_i) / (lc_hi - lc_lo), 0, 1)   in [0, 1].

    A single floor covers both single-residue and distributed/repeat collapse. Because it
    is clamped to zero above ``lc_hi``, it exerts *no force on the healthy distribution*
    (base and passing Complexa both sit above ``lc_hi``) and drives ``g_i -> 1`` for a
    homopolymer. It enters the reward as ``-w_seq_complexity * g_i`` — an absolute
    per-design floor, complementing the between-design (mean-centered) Hamming term, which
    is blind to a group of individually degenerate but mutually *different* modes.

    Parameters
    ----------
    seqs : list[str]
        Decoded binder sequences in the GRPO group.
    lc_hi, lc_lo : float, optional
        Upper (penalty starts below this) and lower (penalty saturates at this) LC edges.
        Defaults ``0.45`` / ``0.15``, calibrated from the measured healthy/collapsed rows.
    kmax : int, optional
        k-mer order passed to :func:`linguistic_complexity`. Defaults to 3.

    Returns
    -------
    tuple[list[float], list[float]]
        ``(penalties, lcs)`` — the per-design penalty ``g_i`` in ``[0, 1]`` and the raw
        ``LC_i`` (for always-on tracking).
    """
    denom = lc_hi - lc_lo
    lcs = [linguistic_complexity(s, kmax) for s in seqs]
    if denom <= 0:
        return [1.0 if lc < lc_hi else 0.0 for lc in lcs], lcs
    pens = [min(1.0, max(0.0, (lc_hi - lc) / denom)) for lc in lcs]
    return pens, lcs


def lc_saturating_reward(seqs: list[str], lc_full: float = 0.7, kmax: int = 3) -> tuple[list[float], list[float]]:
    """Per-design saturating linguistic-complexity **reward** ``r_i`` and the raw ``LC_i``.

    The positive-reward dual of :func:`lc_floor_penalty`. Rather than penalizing
    collapse below a band, this grants full credit once a design is complex enough and
    ramps that credit down toward zero as it collapses — a single saturating ramp:

        r_i = clip(LC_i / lc_full, 0, 1)   in [0, 1].

    ``r_i = 1`` for any ``LC_i >= lc_full`` (no gradient pressure on already-complex
    designs — the reward saturates so the policy is not pushed to pathological
    over-diversification) and ramps linearly to ``0`` at full collapse ``LC_i = 0``. It
    enters the reward as ``+w_seq_complexity * r_i``.

    Parameters
    ----------
    seqs : list[str]
        Decoded binder sequences in the GRPO group.
    lc_full : float, optional
        LC value at (and above) which the reward saturates to ``1``. Defaults ``0.7``.
    kmax : int, optional
        k-mer order passed to :func:`linguistic_complexity`. Defaults to 3.

    Returns
    -------
    tuple[list[float], list[float]]
        ``(rewards, lcs)`` — the per-design reward ``r_i`` in ``[0, 1]`` and the raw
        ``LC_i`` (for always-on tracking).
    """
    lcs = [linguistic_complexity(s, kmax) for s in seqs]
    if lc_full <= 0:
        return [1.0 for _ in lcs], lcs
    rews = [min(1.0, max(0.0, lc / lc_full)) for lc in lcs]
    return rews, lcs


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
