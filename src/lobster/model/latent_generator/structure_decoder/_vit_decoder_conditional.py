"""Decoder variant for flow-matching: tokens + noisy coords + time -> clean coords.

Sibling of :class:`ViTDecoder`. The original is left UNTOUCHED so the live
Tokenizer3diInput SLURM job can preempt/resume mid-build without state_dict
drift. This new class is only ever instantiated from the flow tokenizer
(:class:`Tokenizer3diInputFlow`).

Surface differences vs. :class:`ViTDecoder`:

- The token embedding is owned at this layer (``self.struc_token_embedding``)
  instead of inside ``TimeCondUViTDecoder``. The underlying ``self.net`` is
  built with ``indexed=False`` and ``struc_token_codebook_size=struc_token_dim``
  so its built-in ``embed_struc_tokens`` becomes a learnable ``nn.Linear(D, D)``
  projection over the pre-summed (token + xt) embedding.
- ``self.coord_in_proj : nn.Linear(n_atoms * 3, struc_token_dim)`` projects the
  flow-matching noisy coords ``xt`` into the token-embedding space; the result
  is added to the token embedding before the transformer.
- ``self.time_embedder`` (NoiseConditioningBlock) lifts a scalar ``time_cond``
  to ``(B, 1, time_cond_dim)`` and is forwarded into the U-ViT, which already
  threads time through its FiLM-gated TimeCondAttention / TimeCondFeedForward
  blocks.
- ``num_registers`` learnable prefix tokens (Proteina-style attention sinks)
  are optionally prepended before the transformer and stripped after the
  output projection. Default ``0`` (off).
- ``use_self_conditioning`` adds a second coord input projection
  ``self.coord_in_proj_selfcond`` whose weights are zero-initialised. When
  the caller supplies ``x_selfcond`` (the previous denoised estimate),
  ``coord_in_proj_selfcond(x_selfcond)`` is added to the input token
  embedding alongside ``coord_in_proj(xt)``. Zero-init means a fresh
  model is byte-identical to a non-self-cond model on the first training
  step, and the self-cond pathway only gets activated as the gradients
  carve it out. Mirrors the ESMFold2 self-conditioning trick analysed
  in the Step F section of the plan.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lobster.model.latent_generator.models.vit._vit_utils import (
    NoiseConditioningBlock,
    TimeCondUViTDecoder,
    expand,
)

from ._decoder import BaseDecoder


class ViTDecoderConditional(BaseDecoder):
    """U-ViT decoder conditioned on (3Di tokens, noisy coords xt, time).

    Parameters mirror :class:`ViTDecoder` for parity with the existing
    Hydra ``decoder_factory`` plumbing.

    Extra parameters
    ----------------
    num_3di_classes : int
        Vocabulary size for the 3Di alphabet (default ``20``). The pad index
        used at masked-out positions and as the CFG-null class is
        ``num_3di_classes`` (one past the legal range), so the token embedding
        table has ``num_3di_classes + 1`` rows.
    time_cond_dim : int
        Sinusoidal feature size used by the ``NoiseConditioningBlock``. The
        downstream U-ViT's TimeCondAttention / TimeCondFeedForward layers are
        built with the same dim so the gating layers see a vector of this size.
    num_registers : int
        Number of learnable prefix tokens prepended before the transformer
        (Proteina-style attention sinks). Default ``0`` (off).
    register_init_scale : float
        Init scale for ``self.registers`` (Proteina uses ``randn(...) / 20``).
    use_self_conditioning : bool
        When ``True``, build a second coord input projection
        ``self.coord_in_proj_selfcond`` (zero-initialised) so the caller
        can pass ``x_selfcond`` (the previous denoised estimate) in
        addition to ``xt``. Default ``False``.
    """

    def __init__(
        self,
        struc_token_codebook_size: int,
        indexed: bool,
        struc_token_dim: int,
        data_fixed_size: int,
        n_atoms: int,
        uvit_n_layers: int,
        uvit_n_heads: int,
        uvit_dim_head: int,
        uvit_position_embedding_type: str,
        uvit_patch_size: int = 1,
        translation_scale: float = 1.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_sequential_to_out: bool = False,
        encode_ligand: bool = False,
        ligand_struc_token_codebook_size: int = 256,
        refinement_module: bool = False,
        num_3di_classes: int = 20,
        time_cond_dim: int = 128,
        num_registers: int = 0,
        register_init_scale: float = 0.05,
        use_self_conditioning: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        if encode_ligand:
            raise NotImplementedError(
                "ViTDecoderConditional does not support ligand encoding; the flow variant is protein-only."
            )

        self.translation_scale = translation_scale
        self.n_atoms = n_atoms
        self.refinement_module = refinement_module
        self.num_3di_classes = num_3di_classes
        self.time_cond_dim = time_cond_dim
        self.num_registers = num_registers
        self.struc_token_dim = struc_token_dim
        self.use_self_conditioning = use_self_conditioning

        self.struc_token_embedding = nn.Embedding(num_3di_classes + 1, struc_token_dim)
        self.coord_in_proj = nn.Linear(n_atoms * 3, struc_token_dim)
        self.time_embedder = NoiseConditioningBlock(time_cond_dim, time_cond_dim)

        if use_self_conditioning:
            # Zero-init: a fresh model is byte-identical to a non-self-cond
            # model on the first step, and the self-cond pathway only gets
            # carved out as the gradients reveal it is useful. Matches the
            # EDM/Karras self-cond init convention.
            self.coord_in_proj_selfcond = nn.Linear(n_atoms * 3, struc_token_dim)
            nn.init.zeros_(self.coord_in_proj_selfcond.weight)
            nn.init.zeros_(self.coord_in_proj_selfcond.bias)
        else:
            self.coord_in_proj_selfcond = None

        if num_registers > 0:
            self.registers = nn.Parameter(torch.randn(num_registers, struc_token_dim) * register_init_scale)
        else:
            self.register_parameter("registers", None)

        self.net = TimeCondUViTDecoder(
            struc_token_codebook_size=struc_token_dim,
            struc_token_dim=struc_token_dim,
            seq_len=data_fixed_size,
            patch_size=uvit_patch_size,
            depth=uvit_n_layers,
            heads=uvit_n_heads,
            dim_head=uvit_dim_head,
            n_atoms=n_atoms,
            time_cond_dim=time_cond_dim,
            position_embedding_type=uvit_position_embedding_type,
            indexed=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
            encode_ligand=False,
            ligand_struc_token_codebook_size=ligand_struc_token_codebook_size,
            refinement_module=refinement_module,
        )

    def preprocess(self, coords: Tensor, mask: Tensor, **kwargs):
        return coords, mask

    def get_output_dim(self):
        return [self.n_atoms, 3]

    def forward(
        self,
        x_quant: Tensor,
        seq_mask: Tensor,
        residue_index: Tensor | None = None,
        xt: Tensor | None = None,
        time_cond: Tensor | None = None,
        x_selfcond: Tensor | None = None,
        **kwargs,
    ):
        if xt is None or time_cond is None:
            raise ValueError(
                "ViTDecoderConditional requires both `xt` (noisy coords) and `time_cond` "
                "(scalar t per sample); flow-matching is the only intended use of this module."
            )

        if x_selfcond is not None and not self.use_self_conditioning:
            # Guard: catches the common mistake of wiring the tokenizer's
            # self-cond flag without enabling it on the decoder side.
            raise ValueError(
                "x_selfcond was passed but use_self_conditioning is False; the decoder "
                "has no `coord_in_proj_selfcond` projection. Set "
                "`use_self_conditioning=True` in the decoder config (or stop passing "
                "x_selfcond from the tokenizer)."
            )

        B, L = x_quant.shape

        if seq_mask is not None:
            seq_mask[torch.isnan(seq_mask)] = 0

        tok_emb = self.struc_token_embedding(x_quant)
        xt_flat = xt.reshape(B, L, self.n_atoms * 3)
        xt_emb = self.coord_in_proj(xt_flat)
        x_emb = tok_emb + xt_emb

        if self.use_self_conditioning and x_selfcond is not None:
            sc_flat = x_selfcond.reshape(B, L, self.n_atoms * 3)
            x_emb = x_emb + self.coord_in_proj_selfcond(sc_flat)

        if self.num_registers > 0:
            reg = self.registers.unsqueeze(0).expand(B, -1, -1)
            x_emb = torch.cat([reg, x_emb], dim=1)
            if seq_mask is not None:
                reg_mask = torch.ones(B, self.num_registers, device=seq_mask.device, dtype=seq_mask.dtype)
                seq_mask = torch.cat([reg_mask, seq_mask], dim=1)
            if residue_index is not None:
                reg_idx = torch.zeros(B, self.num_registers, device=residue_index.device, dtype=residue_index.dtype)
                residue_index = torch.cat([reg_idx, residue_index], dim=1)

        t_emb = self.time_embedder(time_cond)

        out = self.net(
            x_emb,
            time_cond=t_emb,
            seq_mask=seq_mask,
            residue_index=residue_index,
        )

        if isinstance(out, dict):
            if "protein_coords_refinement" in out:
                emb_refinement = out["protein_coords_refinement"]
                emb = out["protein_coords"]
                if self.num_registers > 0:
                    emb = emb[:, self.num_registers :]
                    emb_refinement = emb_refinement[:, self.num_registers :]
                    seq_mask_trim = seq_mask[:, self.num_registers :] if seq_mask is not None else None
                else:
                    seq_mask_trim = seq_mask
                assert not torch.isnan(emb).any()
                assert not torch.isnan(emb_refinement).any()
                if seq_mask_trim is not None:
                    emb = emb * expand(seq_mask_trim, emb)
                    emb_refinement = emb_refinement * expand(seq_mask_trim, emb_refinement)
                return {"protein_coords": emb, "protein_coords_refinement": emb_refinement}
            emb = out.get("protein_coords", out)
        else:
            emb = out

        if self.num_registers > 0:
            emb = emb[:, self.num_registers :]
            seq_mask_trim = seq_mask[:, self.num_registers :] if seq_mask is not None else None
        else:
            seq_mask_trim = seq_mask

        assert not torch.isnan(emb).any()
        if seq_mask_trim is not None:
            emb = emb * expand(seq_mask_trim, emb)

        return emb
