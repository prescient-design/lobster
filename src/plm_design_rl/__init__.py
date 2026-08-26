"""``plm_design_rl`` — a self-contained, model-agnostic protein-design RL package.

Reusable building blocks for RL post-training of protein-/biological-sequence design
models: reward terms and oracles (:mod:`plm_design_rl.rewards`), differentiable
log-probability kernels for the policy ratio (:mod:`plm_design_rl.logprob`), and the
out-of-process reward worker-pool clients (:mod:`plm_design_rl.pool`).

The subpackages are pure (no ``trl`` / trainer dependency) so a policy can import the
reward terms and log-prob kernels without pulling in the reward oracles' heavy deps.
The LeFlur GRPO trainer in :mod:`lobster.rl_training` consumes these; the same pieces
are meant to be reused to post-train other iterative sequence-generation policies.
"""

from . import logprob, pool, rewards
from .logprob import dfm_step_kl, dfm_step_logprob, dfm_step_prob
from .pool import (
    ProtenixRewardClient,
    ShapeRewardClient,
    atomic_write_json,
    ensure_queue,
    reward_from_shape,
)

__all__ = [
    "logprob",
    "pool",
    "rewards",
    "dfm_step_prob",
    "dfm_step_logprob",
    "dfm_step_kl",
    "ShapeRewardClient",
    "reward_from_shape",
    "ProtenixRewardClient",
    "atomic_write_json",
    "ensure_queue",
]
