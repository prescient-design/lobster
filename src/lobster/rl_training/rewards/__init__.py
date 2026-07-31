"""Reward terms for the LeFlur GRPO binder policy.

This subpackage collects the reward oracles and shaping terms used by the LeFlur
GRPO trainer, kept separate from the trainer loop and the UME/TRL utilities. The
final reward is a sum of four terms, each a weighted, per-metric-clipped linear
combination (see ``README.md``):

* :mod:`._protenix_reward` — Protenix co-folding confidence reward client + queue
  protocol (weighted linear combo of ptm/iptm/abag_iptm/plddt/gpde/pae metrics →
  scalar reward and the binder pass criterion),
* :mod:`._structure_reward` — self-consistency TM-scores (Kabsch + TM-score) of the
  LeFlur-generated backbone vs the Protenix-predicted backbone (binder + complex),
* :mod:`._diversity_reward` — within-group k-mer-Jaccard novelty on the AA sequence
  and the 3Di structural-token string.

Each module is pure (no ``trl`` dependency) so the policy side can import the
reward terms without pulling in the reward oracle's heavy deps.
"""

from ._protenix_reward import (
    DEFAULT_CONF_WEIGHTS,
    ProtenixRewardClient,
    confidence_components,
    continuous_ip,
    passes,
    reward_from_confidence,
)
from ._structure_reward import kabsch, structure_terms, tm_score
from ._diversity_reward import jaccard_novelty_group, kmer_jaccard

__all__ = [
    "DEFAULT_CONF_WEIGHTS",
    "ProtenixRewardClient",
    "confidence_components",
    "continuous_ip",
    "passes",
    "reward_from_confidence",
    "kabsch",
    "structure_terms",
    "tm_score",
    "jaccard_novelty_group",
    "kmer_jaccard",
]
