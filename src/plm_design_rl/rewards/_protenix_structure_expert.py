"""Structure-expert producer for the Protenix fold-consistency distillation reward.

This is the *structural dual* of the CHORD sequence SFT. Where CHORD folds the reward
signal back through a **sequence expert** — LigandMPNN redesigns the policy's *structure*
into a coherent sequence ``a*`` that becomes the SFT target for the sequence track — this
module produces a **structure expert**: Protenix folds the policy's *output sequence* into
a coherent structure ``X*``, from which we derive BOTH discrete structure representations
the policy generates:

* ``tri_tokens`` (τ*) — the Foldseek 3Di structural alphabet (0..19), via the frozen
  mini3di VAE encoder (:class:`lobster.model.latent_generator.utils.mini3di.Encoder`);
* ``structure_tokens`` (s*) — the LatentGenerator (LG) backbone-geometry codec tokens, via
  the policy's own FSQ codec (``model.encode_structure`` → argmax).

Both are derived from the *same* Protenix output structure, exactly as the user specified:
"derive the 3di tokens and the lg tokens from protenix's outputted structure". The tokens
then become the per-position expert targets for a structure-track SFT distillation
(:func:`lobster.rl_training._structure_sft.structure_sft_loss`), mirroring how the
LigandMPNN sequence becomes ``target_ids`` for the sequence-track CHORD SFT.

Motivation. On the eval sweeps the policy's *binder monomer fold* is reproduced well by
Protenix (target-aligned CA-RMSD ~1.8-2.4 Å) but the *docked pose* disagrees strongly
(binder CA-RMSD ~25-35 Å after target superposition) — Protenix re-docks the sequence
elsewhere. No backbone / pack / 3Di-histogram reward moves docking-pose fidelity because
none of them condition the *policy's own structure endpoint* on a fold that is consistent
with the sequence it emitted. Distilling the policy's structure + 3Di endpoints toward the
Protenix-derived (s*, τ*) closes that sequence↔structure loop directly.

Protenix is used here strictly as a **structure oracle** (coordinates → tokens). Its
confidence outputs (pTM/ipTM/pLDDT) are NOT consumed by this reward — consistent with the
standing rule that Protenix confidence stays an offline label, never a training signal.

Notes
-----
* 3Di is encoded on the **whole complex** by default (``per_chain=False``), matching the
  training-time :class:`~lobster.transforms.Structure3diTransform` default, so an interface
  residue's 3Di state can draw its nearest-neighbour partner from the antigen chain and thus
  encodes inter-chain geometry — precisely the docking signal we want to distil.
* The supervised scope defaults to the **whole complex** (antigen + binder). This is safe
  and correct in both rollout modes because the structure-SFT loss gates every supervised
  position by the per-track *generation* mask (``sup & gen_mask_struc`` in
  :meth:`LeFlurSequenceStructureEncoderLightningModule._sft_step_ce`): a position only ever
  contributes CE if it was actually generated on that track. So with the antigen structure
  pinned (default binder-design rollout, ``inpainting_mask_structure = 0`` at the antigen)
  the whole-complex scope is a no-op on the antigen, and it begins supervising the antigen
  structure/3Di endpoints the moment the antigen structure is itself generated (e.g.
  template-target mode, ``mask_structure[antigen] = 1``). Pass ``supervise_scope="binder"``
  to restrict supervision to the binder chain regardless of the generation mask.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# 3Di invalid-state sentinel (chain termini / masked residues) used by the mini3di encoder.
INVALID_3DI_STATE = 2
SFT_IGNORE_INDEX = -100  # matches lobster.rl_training.rewards._aar_reward.SFT_IGNORE_INDEX

_ENCODER = None  # cached mini3di Encoder (loads frozen kerasify VAE weights once)


def _mini3di_encoder():
    global _ENCODER
    if _ENCODER is None:
        from lobster.model.latent_generator.utils.mini3di import Encoder

        _ENCODER = Encoder()
    return _ENCODER


def derive_3di_tokens(coords_res: np.ndarray | Tensor) -> np.ndarray:
    """Encode a backbone into per-residue 3Di state indices (τ*).

    Parameters
    ----------
    coords_res : ndarray | Tensor
        ``(L, 3, 3)`` backbone coordinates, atom axis ordered ``(N, CA, C)``.

    Returns
    -------
    ndarray
        ``(L,)`` int64 3Di state indices in ``[0, 19]``; chain termini / undefined
        positions carry :data:`INVALID_3DI_STATE`.
    """
    from lobster.model.latent_generator.utils.mini3di import calculate_cb

    coords = torch.as_tensor(np.asarray(coords_res), dtype=torch.float32)
    if coords.ndim != 3 or coords.shape[-2:] != (3, 3):
        raise ValueError(f"coords_res must be (L,3,3); got {tuple(coords.shape)}")
    Ca, Cb, N, C = calculate_cb({"coords_res": coords})
    out = _mini3di_encoder().encode_atoms(Ca, Cb, N, C)
    # masked_array: termini are masked -> fill with INVALID_3DI_STATE.
    states = np.ma.asarray(out["states"]).filled(INVALID_3DI_STATE)
    return states.astype(np.int64)


def derive_structure_tokens(
    encode_structure_fn,
    coords_res: np.ndarray | Tensor,
    *,
    residue_index: np.ndarray | Tensor | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Encode a backbone into LG structure-codec tokens (s*) via the policy's FSQ codec.

    Parameters
    ----------
    encode_structure_fn : Callable
        The policy's ``model.encode_structure(x_gt, mask, residue_index)`` bound method (or
        any callable with that signature). Returns ``(x_quant, x_quant_emb, mask)`` where
        ``x_quant`` is ``(B, L, codebook)`` one-hot-like; tokens are ``argmax(-1)``. Passing
        it as a callable keeps this function unit-testable with a fake codec.
    coords_res : ndarray | Tensor
        ``(L, 3, 3)`` backbone coordinates ``(N, CA, C)``.
    residue_index : ndarray | Tensor | None
        ``(L,)`` residue indices; defaults to ``arange(L)``.
    device : torch.device | str | None
        Device to build the codec inputs on. Defaults to CPU.

    Returns
    -------
    ndarray
        ``(L,)`` int64 structure-token ids (FSQ codebook indices in ``[0, n_tokens)``).
    """
    coords = torch.as_tensor(np.asarray(coords_res), dtype=torch.float32)
    if coords.ndim != 3 or coords.shape[-2:] != (3, 3):
        raise ValueError(f"coords_res must be (L,3,3); got {tuple(coords.shape)}")
    L = coords.shape[0]
    dev = torch.device(device) if device is not None else coords.device
    x_gt = coords.to(dev).unsqueeze(0)  # (1, L, 3, 3)
    # Float (not bool) mask: the LG codec's ViT attention forms attn_mask via an einsum and then
    # does ``sim -= (1 - attn_mask) * 1e6`` — a subtraction that raises on a bool tensor. Every
    # real call site feeds a float padding mask (``valid.to(self.dtype)``); mirror that here so the
    # codec actually runs (a bool mask silently drops the whole struct-SFT expert, scored_frac=0).
    mask = torch.ones(1, L, dtype=x_gt.dtype, device=dev)
    if residue_index is None:
        ridx = torch.arange(L, device=dev).unsqueeze(0)
    else:
        ridx = torch.as_tensor(np.asarray(residue_index), dtype=torch.long, device=dev).view(1, L)
    x_quant, _, _ = encode_structure_fn(x_gt, mask, ridx)
    tokens = torch.argmax(x_quant, dim=-1).squeeze(0)  # (L,)
    return tokens.detach().cpu().numpy().astype(np.int64)


def _pick_binder_chain(chains: np.ndarray, binder_chain: str | None) -> str:
    uniq = list(dict.fromkeys(np.asarray(chains).tolist()))
    if binder_chain is not None:
        if binder_chain not in uniq:
            raise ValueError(f"binder_chain {binder_chain!r} not in {uniq}")
        return binder_chain
    # Convention (matches _tier0_compute.design_hists): chain 'B' is the binder, else last chain.
    return "B" if "B" in uniq else uniq[-1]


def assemble_structure_expert(
    coords_res: np.ndarray,
    chains: np.ndarray,
    aa1: np.ndarray | None = None,
    *,
    encode_structure_fn=None,
    binder_chain: str | None = None,
    residue_index: np.ndarray | None = None,
    device: torch.device | str | None = None,
    supervise_scope: str = "complex",
) -> dict:
    """Build the structure-expert token targets from a parsed complex structure.

    Derives whole-complex 3Di (τ*) and — when ``encode_structure_fn`` is provided — the LG
    structure tokens (s*), then builds the ``supervise_mask`` over the requested scope
    (whole complex by default; see ``supervise_scope``).

    Parameters
    ----------
    coords_res : ndarray
        ``(L, 3, 3)`` backbone ``(N, CA, C)`` for the whole complex (antigen + binder).
    chains : ndarray
        ``(L,)`` chain letters aligned to ``coords_res``.
    aa1 : ndarray | None
        ``(L,)`` one-letter residue names (carried through for downstream alignment/debug).
    encode_structure_fn : Callable | None
        Policy FSQ codec ``encode_structure`` (see :func:`derive_structure_tokens`). If
        ``None``, ``structure_tokens`` is omitted (3Di-only expert).
    binder_chain : str | None
        Binder chain letter; auto-detected ('B' else last chain) when ``None``.
    residue_index : ndarray | None
        Optional ``(L,)`` residue index for the codec; defaults to ``arange(L)``.
    device : torch.device | str | None
        Device for codec inputs.
    supervise_scope : str
        Which positions ``supervise_mask`` covers: ``"complex"`` (default) supervises the
        whole complex (antigen + binder); ``"binder"`` restricts to the binder chain. The
        whole-complex default is safe in both rollout modes — the structure-SFT loss gates
        every supervised position by the per-track generation mask, so pinned-antigen
        positions never contribute CE (see module Notes).

    Returns
    -------
    dict
        ``{"binder_chain", "binder_mask" (L,), "supervise_mask" (L,), "supervise_scope",
        "tri_tokens" (L,), "structure_tokens" (L,)|None, "aa1", "chains", "n_binder"}``.
        Token arrays span the whole complex; use ``supervise_mask`` to select supervised
        positions. Antigen token positions carry valid (non-ignore) targets so they can be
        supervised whenever the antigen structure is generated; the generation-mask gate in
        the loss decides whether they actually contribute.
    """
    coords_res = np.asarray(coords_res, dtype=np.float32)
    chains = np.asarray(chains)
    L = coords_res.shape[0]
    binder = _pick_binder_chain(chains, binder_chain)
    binder_mask = chains == binder

    if supervise_scope == "complex":
        supervise_mask = np.ones(L, dtype=bool)
    elif supervise_scope == "binder":
        supervise_mask = binder_mask
    else:
        raise ValueError(f"supervise_scope must be 'complex' or 'binder'; got {supervise_scope!r}")

    tri_tokens = derive_3di_tokens(coords_res)  # whole complex (captures inter-chain geometry)

    structure_tokens = None
    if encode_structure_fn is not None:
        structure_tokens = derive_structure_tokens(
            encode_structure_fn, coords_res, residue_index=residue_index, device=device
        )

    return {
        "binder_chain": binder,
        "binder_mask": binder_mask,
        "supervise_mask": supervise_mask,
        "supervise_scope": supervise_scope,
        "tri_tokens": tri_tokens,
        "structure_tokens": structure_tokens,
        "aa1": None if aa1 is None else np.asarray(aa1),
        "chains": chains,
        "n_binder": int(binder_mask.sum()),
        "length": L,
    }


def build_struct_sft_targets(
    experts: list[dict | None],
    binder_idx: np.ndarray,
    antigen_idx: np.ndarray,
    G: int,
    L: int,
) -> dict:
    """Scatter per-design structure-expert tokens onto the padded ``(G, L)`` layout.

    The trainer-side bridge from :func:`assemble_structure_expert` outputs (one per design,
    tokens in whole-complex **antigen-then-binder** order — the same order the sctm reward
    uses, ``concatenate([antigen, binder])``) to the dense ``(G, L)`` target tensors the
    structure-SFT loss (:func:`lobster.rl_training._structure_sft.structure_sft_loss`)
    consumes. Kept pure (numpy only) so the antigen/binder scatter is unit-testable without a
    trainer or a folded structure.

    Parameters
    ----------
    experts : list[dict | None]
        Length-``G`` list of :func:`assemble_structure_expert` results, or ``None`` for a
        design the fold/parse failed on (→ an all-ignore row, nothing supervised). Each
        expert's token arrays must span ``len(antigen_idx) + len(binder_idx)`` positions in
        antigen-then-binder order (the caller validates the predicted chain lengths and passes
        ``None`` on a mismatch).
    binder_idx, antigen_idx : ndarray
        Full-layout position indices (``np.nonzero`` order) of the binder and antigen
        residues, matching the chain order the expert tokens are laid out in.
    G, L : int
        Group size and padded layout length.

    Returns
    -------
    dict
        ``{"struct_target_ids" (G, L) int64, "tri_target_ids" (G, L) int64,
        "supervise_mask" (G, L) bool, "n_scored" int}``. Unsupervised / failed positions
        carry ``SFT_IGNORE_INDEX`` (targets) and ``False`` (mask). ``struct_target_ids`` is
        all-ignore when the expert carries no LG structure tokens (3Di-only expert).
    """
    binder_idx = np.asarray(binder_idx, dtype=np.int64)
    antigen_idx = np.asarray(antigen_idx, dtype=np.int64)
    na, nb = int(antigen_idx.shape[0]), int(binder_idx.shape[0])
    struct_target_ids = np.full((G, L), SFT_IGNORE_INDEX, dtype=np.int64)
    tri_target_ids = np.full((G, L), SFT_IGNORE_INDEX, dtype=np.int64)
    supervise_mask = np.zeros((G, L), dtype=bool)
    n_scored = 0
    for i, ex in enumerate(experts):
        if ex is None:
            continue
        tri = np.asarray(ex["tri_tokens"], dtype=np.int64)
        sup = np.asarray(ex["supervise_mask"], dtype=bool)
        if tri.shape[0] != na + nb or sup.shape[0] != na + nb:
            continue  # length mismatch vs the padded layout — skip (all-ignore row)
        tri_target_ids[i, antigen_idx] = tri[:na]
        tri_target_ids[i, binder_idx] = tri[na:]
        supervise_mask[i, antigen_idx] = sup[:na]
        supervise_mask[i, binder_idx] = sup[na:]
        st = ex.get("structure_tokens")
        if st is not None:
            st = np.asarray(st, dtype=np.int64)
            if st.shape[0] == na + nb:
                struct_target_ids[i, antigen_idx] = st[:na]
                struct_target_ids[i, binder_idx] = st[na:]
        n_scored += 1
    return {
        "struct_target_ids": struct_target_ids,
        "tri_target_ids": tri_target_ids,
        "supervise_mask": supervise_mask,
        "n_scored": n_scored,
    }


def structure_expert_from_cif(
    cif_path: str,
    *,
    encode_structure_fn=None,
    binder_chain: str | None = None,
    device: torch.device | str | None = None,
    supervise_scope: str = "complex",
) -> dict | None:
    """Fold-branch expert: parse a Protenix output cif into (s*, τ*) structure-expert tokens.

    Parses ``cif_path`` (a Protenix-predicted structure of the policy's output *sequence*)
    into an ``(N, CA, C)`` backbone, then derives the 3Di and LG structure tokens for the
    binder. Returns ``None`` if the cif cannot be parsed or the binder chain is too short.

    Parameters
    ----------
    cif_path : str
        Path to a Protenix ``*.cif`` structure prediction.
    encode_structure_fn : Callable | None
        Policy FSQ codec ``encode_structure``; if ``None``, 3Di-only.
    binder_chain : str | None
        Binder chain letter (auto-detected when ``None``).
    device : torch.device | str | None
        Device for codec inputs.
    supervise_scope : str
        Supervised scope for ``supervise_mask``; ``"complex"`` (default) or ``"binder"``.
        See :func:`assemble_structure_expert`.

    Returns
    -------
    dict | None
        Same schema as :func:`assemble_structure_expert`, plus ``"cif_path"``; or ``None``.
    """
    import os
    import sys

    # parse_cif lives with the offline scoring tools; add its dir to path lazily.
    _scripts = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
        "scripts",
    )
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    import _tier0_compute as C  # type: ignore  # noqa: E402

    parsed = C.parse_cif(cif_path)
    if parsed is None:
        return None
    coords, chains, aa1 = parsed
    try:
        expert = assemble_structure_expert(
            coords,
            chains,
            aa1,
            encode_structure_fn=encode_structure_fn,
            binder_chain=binder_chain,
            device=device,
            supervise_scope=supervise_scope,
        )
    except ValueError:
        return None
    if expert["n_binder"] < 4:
        return None
    expert["cif_path"] = cif_path
    return expert
