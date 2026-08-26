"""Protenix fold-consistency SFT: the structural dual of the CHORD sequence SFT.

Distills the policy's **structure-track** (LG codec, ``structure_tokens``) and **3Di-track**
(``tri_tokens``) endpoint predictions toward a *structure expert* derived by folding the
policy's own output sequence with Protenix (see
:mod:`lobster.rl_training.rewards._protenix_structure_expert`). It is the exact mirror of
:meth:`LeFlurSequenceStructureEncoderLightningModule.sequence_sft_loss` — same φ-weighted,
mask-normalized, expert-context CE — applied to the structure and 3Di tracks instead of the
sequence track.

Implemented as a standalone function (rather than a new model method) for the prototype so
it can be iterated and unit-tested in isolation; it reuses the model's already-tested pure
helpers ``_iter_traj_steps``, ``_expert_context_seq`` (token-agnostic) and ``_sft_step_ce``.
Promotion to a model method + trainer wiring (``struct_sft_mu``, ``struct_target_ids`` /
``tri_target_ids`` payload) is the follow-up once the reward is validated.

Expert-context rationale (mirrors CHORD). The Protenix-derived tokens ``(s*, τ*)`` form a
mutually-coherent structure. Conditioning the SFT forward on the policy's *own* rollout
tokens at revealed positions would fit ``p_θ(s*_masked | policy context)`` — a chimera. So
:func:`_expert_context_seq` overwrites revealed supervised positions with the expert token,
distilling ``p_θ(s* | s*-context)`` — pure denoising toward the coherent folded structure.
The φ weight ``p_t(1-p_t)`` then auto-suppresses positions the policy already agrees on. With
``supervise_scope="complex"`` this runs over both the antigen and the binder structures.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _structure_endpoint_logits(
    model,
    xt_dev: dict,
    t_seq: Tensor,
    t_struc: Tensor,
    static: dict,
    *,
    struct_override: Tensor | None,
    tri_override: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Raw (conditional, unbiased) structure + 3Di endpoint logits for one rollout step.

    Mirrors :meth:`_sft_seq_endpoint_logits`: a plain ``forward`` with NO classifier-free
    guidance and NO sampler logit-bias / diversity penalty (those are sampling tricks, not
    part of the learned conditional the SFT term shapes). ``*_override`` swap in the expert
    context on the respective track. Returns ``(structure_logits, tri_logits | None)``.
    """
    xt = dict(xt_dev)
    if struct_override is not None:
        xt["structure_tokens"] = struct_override
    if tri_override is not None and "tri_tokens" in xt:
        xt["tri_tokens"] = tri_override
    out = model.forward(
        xt,
        static["mask"],
        static["residue_index"],
        static["conditioning_tensor"],
        timesteps={"sequence_tokens": t_seq, "structure_tokens": t_struc},
        chain_ids=static["chain_ids"],
        template_structure_tokens=static["template_structure_tokens"],
        scalar_cond_bins=static["scalar_cond_bins"],
    )
    return out["structure_logits"], out.get("tri_logits")


def structure_sft_loss(
    model,
    trajectory: dict,
    struct_target: Tensor | None,
    tri_target: Tensor | None,
    supervise_mask: Tensor,
    *,
    w_struct: float = 1.0,
    w_tri: float = 1.0,
    step_indices: Sequence[int] | None = None,
    masked_only: bool = True,
    use_phi: bool = True,
    row_mask: Tensor | None = None,
    grad_checkpoint: bool = False,
) -> Tensor:
    """Fold-consistency SFT loss over the structure + 3Di tracks.

    The plain CHORD-style distillation, applied over the **whole Protenix complex** (both the
    antigen and the binder) when ``supervise_mask`` spans both chains (the default
    ``supervise_scope="complex"`` from the expert producer). Per rollout step, revealed supervised
    positions are overwritten with the expert token (:meth:`_expert_context_seq`) so the forward
    conditions on the coherent Protenix structure, and CE is taken at the supervised positions,
    φ-weighted by ``p_t(1-p_t)``. Same formulation on both the structure (LG codec) and 3Di tracks.

    Parameters
    ----------
    model : LeFlurSequenceStructureEncoderLightningModule
        The policy (provides ``forward``, ``_iter_traj_steps``, ``_expert_context_seq``,
        ``_sft_step_ce``).
    trajectory : dict
        Rollout store from ``rollout_with_logprobs`` (same object the GRPO log-prob path
        consumes).
    struct_target : Tensor | None
        ``(B, L)`` expert LG structure-codec tokens ``s*`` (FSQ ids), ``< 0`` = ignore.
        ``None`` disables the structure track.
    tri_target : Tensor | None
        ``(B, L)`` expert 3Di tokens ``τ*`` in ``[0, 19]``, ``< 0`` = ignore. ``None`` (or an
        inactive 3Di track) disables the 3Di track.
    supervise_mask : Tensor
        ``(B, L)`` boolean mask of supervised positions — the whole complex (antigen + binder) by
        default (see ``supervise_scope`` in the expert producer).
    w_struct, w_tri : float
        Per-track blend weights inside this loss.
    step_indices : Sequence[int] | None
        Which rollout steps to sum over (``None`` = all).
    masked_only : bool
        Restrict supervision at each step to positions still masked on that track.
    use_phi : bool
        Apply the CHORD ``φ = p_t(1-p_t)`` detached token weighting.
    row_mask : Tensor | None
        Optional ``(B,)`` mask zeroing whole designs (e.g. a positive-advantage reward gate).
    grad_checkpoint : bool
        Checkpoint each step's forward to bound peak memory.

    Returns
    -------
    Tensor
        Scalar SFT loss (mean over supervised steps of the summed per-track φ-weighted CE).
        Zero when nothing is supervised.
    """
    static = trajectory["static"]
    device = static["mask"].device
    sup = supervise_mask.to(device=device).bool()
    if row_mask is not None:
        sup = sup & row_mask.to(device=device).bool().unsqueeze(-1)

    mask_index_struc = static["mask_index_struc"]
    mask_index_tri = static["mask_index_tri"]
    gen_mask_struc = static["gen_mask_struc"].to(device).bool()

    do_struct = struct_target is not None and w_struct > 0
    struct_tgt = struct_target.to(device=device, dtype=torch.long) if do_struct else None

    gm_tri = static.get("gen_mask_tri")
    do_tri = bool(static.get("use_3di_track")) and gm_tri is not None and tri_target is not None and w_tri > 0
    if do_tri:
        gen_mask_tri = gm_tri.to(device).bool()
        tri_tgt = tri_target.to(device=device, dtype=torch.long)

    use_ckpt = grad_checkpoint and torch.is_grad_enabled()
    total: Tensor | None = None
    n_steps = 0
    for rec, xt_dev, t_seq, t_struc in model._iter_traj_steps(trajectory, step_indices):
        ctx_struct = (
            model._expert_context_seq(xt_dev["structure_tokens"], struct_tgt, mask_index_struc) if do_struct else None
        )
        ctx_tri = (
            model._expert_context_seq(xt_dev["tri_tokens"], tri_tgt, mask_index_tri)
            if do_tri and "tri_tokens" in xt_dev
            else None
        )
        if use_ckpt:
            s_logits, tri_logits = torch.utils.checkpoint.checkpoint(
                _structure_endpoint_logits,
                model,
                xt_dev,
                t_seq,
                t_struc,
                static,
                struct_override=ctx_struct,
                tri_override=ctx_tri,
                use_reentrant=False,
            )
        else:
            s_logits, tri_logits = _structure_endpoint_logits(
                model, xt_dev, t_seq, t_struc, static, struct_override=ctx_struct, tri_override=ctx_tri
            )

        step: Tensor | None = None
        if ctx_struct is not None:
            ce_s = model._sft_step_ce(
                s_logits,
                struct_tgt,
                sup,
                ctx_struct,
                gen_mask_struc,
                mask_index_struc,
                label="hard",
                soft_targets=None,
                temperature=1.0,
                masked_only=masked_only,
                use_phi=use_phi,
            )
            step = w_struct * ce_s
        if ctx_tri is not None and tri_logits is not None:
            ce_t = model._sft_step_ce(
                tri_logits,
                tri_tgt,
                sup,
                ctx_tri,
                gen_mask_tri,
                mask_index_tri,
                label="hard",
                soft_targets=None,
                temperature=1.0,
                masked_only=masked_only,
                use_phi=use_phi,
            )
            step = (w_tri * ce_t) if step is None else (step + w_tri * ce_t)
        if step is None:
            continue
        total = step if total is None else total + step
        n_steps += 1

    if total is None or n_steps == 0:
        return torch.zeros((), device=device)
    return total / n_steps
