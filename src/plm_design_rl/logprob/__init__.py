"""Differentiable log-probability kernels for RL over sequence-generation policies.

Model-agnostic building blocks for recomputing per-step trajectory log-probs (and the
per-step KL) with gradient, used by the GRPO trainer to form the policy-ratio objective.

* :mod:`.dfm` — discrete-flow-matching (DFM) single-step probability / log-prob / KL
  kernels (torch-only), for masked/flow-based iterative generators such as LeFlur.
"""

from .dfm import dfm_step_kl, dfm_step_logprob, dfm_step_prob

__all__ = [
    "dfm_step_prob",
    "dfm_step_logprob",
    "dfm_step_kl",
]
