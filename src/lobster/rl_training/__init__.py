"""
Reinforcement Learning training utilities for lobster.

This module provides utilities for training models using reinforcement learning
techniques: a TRL-based text-completion GRPO stack for UME reward models, and a
GRPO stack for the LeFlur absorbing-state discrete flow-matching binder policy.

The LeFlur GRPO helpers (e.g. :mod:`lobster.rl_training._dfm_logprob`) are pure
PyTorch and have no ``trl`` dependency; the UME TRL trainers are imported lazily
so the LeFlur path is usable in environments without ``trl`` installed.
"""

from .reward_functions import UMERewardFunction, create_ume_reward_wrapper

# The differentiable DFM step-kernel log-prob helpers underpin the LeFlur GRPO
# trainer and depend only on torch / bionemo — always available.
from ._dfm_logprob import dfm_step_kl, dfm_step_logprob, dfm_step_prob

# Reward oracles + shaping terms for the LeFlur GRPO policy live in the ``rewards``
# subpackage: the Protenix co-folding confidence client (weighted-linear combo), the
# structure self-consistency TM-score, and within-group Jaccard novelty. All pure
# python / numpy / torch (no trl), re-exported here for convenience.
from .rewards import (
    DEFAULT_CONF_WEIGHTS,
    ProtenixRewardClient,
    confidence_components,
    continuous_ip,
    hamming_novelty_group,
    jaccard_novelty_group,
    kmer_jaccard,
    lc_floor_penalty,
    linguistic_complexity,
    passes,
    reward_from_confidence,
    structure_terms,
)

# The standalone LeFlur GRPO trainer — pure torch (no trl); composes the reward
# client, diversity terms, and the policy's trajectory log-prob/KL kernels.
from ._leflur_grpo_trainer import (
    GRPOTrainerConfig,
    LeFlurGRPOTrainer,
    TargetSpec,
)

__all__ = [
    "UMERewardFunction",
    "create_ume_reward_wrapper",
    "create_ume_grpo_trainer",
    "train_ume_grpo",
    "dfm_step_prob",
    "dfm_step_logprob",
    "dfm_step_kl",
    "DEFAULT_CONF_WEIGHTS",
    "ProtenixRewardClient",
    "confidence_components",
    "continuous_ip",
    "hamming_novelty_group",
    "jaccard_novelty_group",
    "kmer_jaccard",
    "lc_floor_penalty",
    "linguistic_complexity",
    "passes",
    "reward_from_confidence",
    "structure_terms",
    "GRPOTrainerConfig",
    "LeFlurGRPOTrainer",
    "TargetSpec",
]


def __getattr__(name: str):
    """Lazily import the TRL-based UME GRPO trainers (optional ``trl`` dependency)."""
    if name in ("create_ume_grpo_trainer", "train_ume_grpo"):
        from .trainers import create_ume_grpo_trainer, train_ume_grpo

        return {"create_ume_grpo_trainer": create_ume_grpo_trainer, "train_ume_grpo": train_ume_grpo}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
