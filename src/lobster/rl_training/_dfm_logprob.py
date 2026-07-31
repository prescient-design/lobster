"""Differentiable, out-of-place reimplementation of the absorbing-state DFM step kernel.

The bionemo ``DiscreteFlowMatcher.step`` transition kernel is fully differentiable
in the model logits given ``(t, dt, xt)`` — the only stochastic operation is the
terminal ``torch.multinomial`` draw. For GRPO we need the *log-probability* of an
already-sampled transition ``xt -> x_next`` under the current policy (and under a
frozen reference), reconstructed with gradient flowing back into the logits.

Two facts about the upstream implementation force a faithful re-write rather than a
call-through (see
``.venv/.../bionemo/moco/interpolants/continuous_time/discrete/discrete_flow_matching.py``):

1. ``step`` (line ~157) does ``x_1_pred_logits[..., mask_index] = -1e9`` where
   ``x_1_pred_logits`` is an *alias* of the caller's ``logits`` — it mutates the
   input tensor in place, which breaks autograd and corrupts a tensor we still
   need for the reference-model pass.
2. ``_regularize_step_probs`` (line ~187) uses in-place ``scatter_`` twice, which
   is not autograd-friendly for repeated backward passes.

This module reproduces the ``use_mask=True`` (absorbing-state / masked-prior) branch
*exactly* — same softmax-at-temperature, same unmask/remask terms, same
final-step ``(t + dt < 1)`` remask gate, same clamp-and-renormalize regularization —
but entirely out-of-place, so the returned ``step_prob`` is differentiable in
``logits`` and the input tensor is never touched.

Notes
-----
Only the masked-prior branch is implemented; every LeFlur track
(sequence / structure / 3Di) uses a ``DiscreteMaskedPrior``, so ``use_mask`` is
always ``True`` in our regime. The uniform-prior branch raises
``NotImplementedError``.

See Also
--------
bionemo.moco.interpolants.continuous_time.discrete.discrete_flow_matching.DiscreteFlowMatcher.step
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "dfm_step_prob",
    "dfm_step_logprob",
    "dfm_step_kl",
]


def _as_time_tensor(value: Tensor | float, batch_size: int, ref: Tensor) -> Tensor:
    """Broadcast a scalar/1-D time-like value to ``(B, 1, 1)`` matching ``ref``.

    Mirrors ``bionemo.moco.interpolants.base_interpolant.pad_like``: a value of
    shape ``(B,)`` becomes ``(B, 1, 1)`` for a rank-3 ``ref``; a Python float or a
    0-d tensor is expanded to the batch dimension first.

    Parameters
    ----------
    value : Tensor | float
        The time ``t`` or step size ``dt``. Either a Python float, a 0-d tensor, or
        a 1-d tensor of shape ``(B,)``.
    batch_size : int
        Batch size ``B`` used to expand scalars.
    ref : Tensor
        Reference tensor whose ``dtype``, ``device`` and rank the result matches.

    Returns
    -------
    Tensor
        A tensor of shape ``(B, 1, ..., 1)`` with ``ref.ndim`` dimensions.
    """
    if not isinstance(value, Tensor):
        value = torch.tensor([float(value)] * batch_size, dtype=ref.dtype, device=ref.device)
    else:
        value = value.to(dtype=ref.dtype, device=ref.device)
        if value.ndim == 0:
            value = value.reshape(1).expand(batch_size)
    # pad trailing singleton dims to match ref rank (value is 1-d of length B here)
    if value.ndim < ref.ndim:
        value = value.view(list(value.shape) + [1] * (ref.ndim - value.ndim))
    return value


def _regularize_step_probs_functional(step_prob: Tensor, xt: Tensor) -> Tensor:
    """Out-of-place equivalent of ``DiscreteFlowMatcher._regularize_step_probs``.

    Clamps the raw step probabilities to ``[0, 1]``, sets the probability of the
    *current* state ``xt`` to zero, then assigns it the remaining probability mass
    so each row sums to one, and clamps once more. The upstream implementation does
    this with two in-place ``scatter_`` calls; here it is done with a one-hot mask
    so gradients flow and the input is not mutated.

    Parameters
    ----------
    step_prob : Tensor
        Raw step probabilities of shape ``(B, L, S)``.
    xt : Tensor
        Current discrete state of shape ``(B, L)``.

    Returns
    -------
    Tensor
        Regularized step probabilities of shape ``(B, L, S)``, each row a valid
        (non-normalized-to-exactly-one only under clamp truncation) distribution.
    """
    num_classes = step_prob.shape[-1]
    step_prob = step_prob.clamp(0.0, 1.0)
    xt_one_hot = F.one_hot(xt.long(), num_classes).to(step_prob.dtype)  # (B, L, S)
    # Zero the current-state column (out-of-place mirror of the first scatter_).
    step_prob = step_prob * (1.0 - xt_one_hot)
    # Assign the remaining mass to the current-state column (mirror of second scatter_).
    remaining = 1.0 - step_prob.sum(dim=-1, keepdim=True)
    step_prob = step_prob + xt_one_hot * remaining
    step_prob = step_prob.clamp(0.0, 1.0)
    return step_prob


def dfm_step_prob(
    logits: Tensor,
    t: Tensor | float,
    dt: Tensor | float,
    xt: Tensor,
    mask_index: int,
    temperature: float = 1.0,
    stochasticity: float = 1.0,
) -> Tensor:
    """Compute the absorbing-state DFM per-step transition distribution, differentiably.

    Reproduces the ``use_mask=True`` branch of
    ``DiscreteFlowMatcher.step`` exactly, but out-of-place: the returned tensor is
    differentiable in ``logits`` and ``logits`` itself is never modified.

    Parameters
    ----------
    logits : Tensor
        Model logits of shape ``(B, L, S)`` (predicted ``x_1`` logits, including the
        mask column at ``mask_index``).
    t : Tensor | float
        Current time. A Python float, 0-d tensor, or ``(B,)`` tensor. Values in
        ``[0, 1)``.
    dt : Tensor | float
        Time-step increment. Same accepted shapes as ``t``.
    xt : Tensor
        Current discrete state of shape ``(B, L)``.
    mask_index : int
        Index of the absorbing (mask) token in the vocabulary; must satisfy
        ``0 <= mask_index < S``.
    temperature : float, optional
        Softmax temperature applied to the logits. Defaults to ``1.0``.
    stochasticity : float, optional
        Stochasticity (noise) level controlling unmask/remask rates. Defaults to
        ``1.0``.

    Returns
    -------
    Tensor
        Per-step transition probabilities ``step_prob`` of shape ``(B, L, S)``.

    Raises
    ------
    ValueError
        If ``mask_index`` is out of range for the logits vocabulary.

    Notes
    -----
    The division by ``(1 - t)`` has no epsilon, matching upstream exactly; callers
    must not pass ``t == 1``. The remask term is gated by ``(t + dt < 1)`` so the
    final step performs no remasking, again matching upstream.
    """
    batch_size, _, num_classes = logits.shape
    if not 0 <= mask_index < num_classes:
        raise ValueError(f"mask_index={mask_index} out of range for vocab size S={num_classes}")

    t_b = _as_time_tensor(t, batch_size, logits)  # (B, 1, 1)
    dt_b = _as_time_tensor(dt, batch_size, logits)  # (B, 1, 1)

    # Out-of-place equivalent of `x_1_pred_logits[..., mask_index] = -1e9`.
    mask_col = torch.zeros(num_classes, dtype=torch.bool, device=logits.device)
    mask_col[mask_index] = True
    masked_logits = logits.masked_fill(mask_col.view(1, 1, num_classes), -1.0e9)

    x1_prob = F.softmax(masked_logits / temperature, dim=-1)  # (B, L, S)

    xt_is_mask = (xt == mask_index).unsqueeze(-1).to(x1_prob.dtype)  # (B, L, 1)
    mask_one_hot = (
        F.one_hot(torch.tensor(mask_index, device=logits.device), num_classes).to(x1_prob.dtype).view(1, 1, num_classes)
    )
    final_gate = (t_b + dt_b < 1).to(x1_prob.dtype)  # (B, 1, 1) — no remask on final step

    step_prob = (
        dt_b * x1_prob * ((1 + stochasticity * t_b) / (1 - t_b)) * xt_is_mask
        + dt_b * (1 - xt_is_mask) * mask_one_hot * stochasticity * final_gate
    )
    step_prob = _regularize_step_probs_functional(step_prob, xt)
    return step_prob


def dfm_step_logprob(
    logits: Tensor,
    t: Tensor | float,
    dt: Tensor | float,
    xt: Tensor,
    x_next: Tensor,
    gen_mask: Tensor,
    mask_index: int,
    temperature: float = 1.0,
    stochasticity: float = 1.0,
    eps: float = 1e-9,
) -> Tensor:
    """Log-probability of a sampled transition ``xt -> x_next`` over generated positions.

    Computes ``sum_{l in gen} log step_prob[b, l, x_next[b, l]]`` for each batch
    element, where ``step_prob`` is the differentiable transition distribution from
    :func:`dfm_step_prob`. Only positions where ``gen_mask`` is true contribute
    (fixed / inpainted positions are excluded).

    Parameters
    ----------
    logits : Tensor
        Model logits of shape ``(B, L, S)``.
    t : Tensor | float
        Current time (see :func:`dfm_step_prob`).
    dt : Tensor | float
        Time-step increment (see :func:`dfm_step_prob`).
    xt : Tensor
        Current discrete state of shape ``(B, L)``.
    x_next : Tensor
        Sampled next state of shape ``(B, L)``.
    gen_mask : Tensor
        Boolean/0-1 mask of shape ``(B, L)`` selecting generated (non-fixed)
        positions.
    mask_index : int
        Index of the absorbing (mask) token.
    temperature : float, optional
        Softmax temperature. Defaults to ``1.0``.
    stochasticity : float, optional
        Stochasticity level. Defaults to ``1.0``.
    eps : float, optional
        Floor added inside the log for numerical stability. Defaults to ``1e-9``.

    Returns
    -------
    Tensor
        Per-batch summed log-probability of shape ``(B,)``, differentiable in
        ``logits``.

    Notes
    -----
    ``torch.multinomial`` (the upstream sampler) treats its input as unnormalized
    weights and divides by the per-row sum, so the *actual* action distribution is
    ``step_prob / step_prob.sum(-1)``. This matters at the final step where the
    ``1/(1-t)`` factor can push a masked row's sum above one; we normalize here so
    the log-prob is unbiased. In the common (well-regularized) case each row
    already sums to one and the normalization is a no-op.
    """
    step_prob = dfm_step_prob(logits, t, dt, xt, mask_index, temperature=temperature, stochasticity=stochasticity)
    row_sum = step_prob.sum(dim=-1).clamp_min(eps)  # (B, L)
    chosen = step_prob.gather(dim=-1, index=x_next.long().unsqueeze(-1)).squeeze(-1)  # (B, L)
    logp = torch.log(chosen.clamp_min(eps)) - torch.log(row_sum)  # (B, L) — normalized categorical
    gen = gen_mask.to(logp.dtype)
    return (logp * gen).sum(dim=-1)  # (B,)


def dfm_step_kl(
    step_prob_theta: Tensor,
    step_prob_ref: Tensor,
    gen_mask: Tensor,
    eps: float = 1e-9,
) -> Tensor:
    """Per-batch categorical KL ``KL(pi_theta || pi_ref)`` summed over generated positions.

    Uses the closed-form categorical KL over the full vocabulary simplex at each
    position (lower variance than a single-sample estimator, cheap because both
    step distributions are already materialized):

    ``sum_l gen[l] * sum_s p_theta[l, s] * (log p_theta[l, s] - log p_ref[l, s])``.

    Parameters
    ----------
    step_prob_theta : Tensor
        Current-policy step probabilities of shape ``(B, L, S)`` (from
        :func:`dfm_step_prob`).
    step_prob_ref : Tensor
        Reference-policy step probabilities of shape ``(B, L, S)``.
    gen_mask : Tensor
        Boolean/0-1 mask of shape ``(B, L)`` selecting generated positions.
    eps : float, optional
        Floor inside the logs for numerical stability. Defaults to ``1e-9``.

    Returns
    -------
    Tensor
        Per-batch summed KL of shape ``(B,)``, differentiable in
        ``step_prob_theta``.

    Notes
    -----
    Both step-prob tensors are the raw (possibly unnormalized) weights returned by
    :func:`dfm_step_prob`; they are row-normalized here so the KL is between the
    true categorical action distributions (see :func:`dfm_step_logprob`).
    """
    p = step_prob_theta / step_prob_theta.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = step_prob_ref / step_prob_ref.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_per_pos = (p * (torch.log(p) - torch.log(q))).sum(dim=-1)  # (B, L)
    gen = gen_mask.to(kl_per_pos.dtype)
    return (kl_per_pos * gen).sum(dim=-1)  # (B,)
