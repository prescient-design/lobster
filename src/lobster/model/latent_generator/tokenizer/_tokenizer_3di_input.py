"""Latent Generator variant — 3Di tokens in, backbone coordinates out.

The canonical :class:`TokenizerMulti` reads N/CA/C coordinates and learns
a discrete bottleneck via an encoder → quantizer → decoder pipeline. This
variant swaps that pipeline for a single embedding lookup on the already-
discrete Foldseek 3Di token sequence (20-class alphabet, computed at
data-load time by
:class:`lobster.transforms._structure_transforms.Structure3diTransform`)
and reuses the existing decoder / loss factories. The model is a
1-pass auto-encoder where the input alphabet is 3Di and the supervision
signal is the ground-truth backbone coordinates.

The training objective therefore is "given the 3Di structural state of
each residue, predict the N/CA/C coordinates" — i.e. *decode* 3Di tokens
to backbones.

This module is intentionally minimal:

- ``encoder`` = ``None`` (no encoder needed — the 3Di state IDs are fed
  directly into the decoder's built-in indexed embedding by setting
  ``indexed=True`` and ``struc_token_codebook_size=num_3di_classes + 1``
  on the wrapped ``ViTDecoder``).
- ``quantizer`` = ``None`` (the input is already discrete).
- ``decoder_factory`` and ``loss_factory`` are reused from the existing
  LG building blocks (the ``vit_decoder`` + ``[l2_loss, pairwise_l2_loss]``
  defaults).

Public surface mirrors :class:`TokenizerMulti` to keep the Hydra wiring
familiar.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import hydra
import lightning.pytorch as pl
import omegaconf
import torch

from lobster.model.latent_generator.structure_decoder import DecoderFactory

from ._loss_factory import LossFactory

logger = logging.getLogger(__name__)

# Standard Foldseek 3Di alphabet (20 states). See `mini3di.Encoder`. The pad
# index follows the same convention used elsewhere in the codebase: an
# extra class one past the vocabulary, never produced by `Structure3diTransform`.
NUM_3DI_CLASSES = 20


class Tokenizer3diInput(pl.LightningModule):
    """3Di-tokens-in / backbone-coords-out Lightning module.

    Parameters
    ----------
    decoder_factory
        A pre-built ``DecoderFactory`` (or a ``DictConfig`` describing one).
        The wrapped ``ViTDecoder`` MUST be configured with ``indexed=True``
        and ``struc_token_codebook_size=num_3di_classes + 1`` so its
        built-in embedding lookup can consume the 3Di state IDs directly
        (no projection layer is added by this class).
    loss_factory
        Reuses the existing ``LossFactory`` (``l2_loss`` + ``pairwise_l2_loss``
        are the canonical pair for coordinate reconstruction).
    optim, lr_scheduler
        Same lifecycle as :class:`TokenizerMulti`.
        num_3di_classes
            Vocabulary size for the 3Di alphabet (default ``20``). The pad
            index used at masked-out positions is ``num_3di_classes`` (one
            past the legal range); the wrapped decoder's embedding table
            must therefore have ``num_3di_classes + 1`` rows.
        num_warmup_steps, num_training_steps
            Forwarded to the LR scheduler factory.
        ckpt_path
            Optional Lightning checkpoint to resume from. Stored on the
            instance and read by ``lobster.cmdline.train`` to forward to
            ``trainer.fit(..., ckpt_path=...)``. Mirrors the
            :class:`TokenizerMulti` wiring.
    """

    def __init__(
        self,
        decoder_factory: Callable[..., DecoderFactory],
        loss_factory: Callable[..., LossFactory],
        optim: Callable[..., torch.optim.Optimizer],
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        num_3di_classes: int = NUM_3DI_CLASSES,
        num_warmup_steps: int = 50_000,
        num_training_steps: int = 500_000,
        automatic_optimization: bool = True,
        ckpt_path: str | None = None,
    ):
        super().__init__()

        if isinstance(decoder_factory, omegaconf.DictConfig):
            decoder_factory = hydra.utils.instantiate(decoder_factory)
        if isinstance(loss_factory, omegaconf.DictConfig):
            loss_factory = hydra.utils.instantiate(loss_factory)
        if isinstance(optim, omegaconf.DictConfig):
            optim = hydra.utils.instantiate(optim)
        if isinstance(lr_scheduler, omegaconf.DictConfig):
            lr_scheduler = hydra.utils.instantiate(lr_scheduler)

        self.decoder_factory = decoder_factory
        self.loss_factory = loss_factory
        self.optim_factory = optim
        self.lr_scheduler = lr_scheduler
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.automatic_optimization = automatic_optimization

        # Match TokenizerMulti's interface for compatibility with inference utilities.
        self.encoder = None
        self.quantizer = None
        self.freeze_decoder = False
        self.freeze_encoder = False
        self.freeze_quantizer = False

        self.num_3di_classes = num_3di_classes
        self.ckpt_path = ckpt_path

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Tokenizer3diInput: total params=%d, trainable=%d", total, trainable)

    # ------------------------------------------------------------------
    # Featurisation: pull 3Di states + mask + residue_index out of the batch.
    # The wrapped decoder owns the embedding lookup (`indexed=True`).
    # ------------------------------------------------------------------

    def featurize(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (states, seq_mask, residue_index) ready for the decoder.

        Expects ``batch`` to carry keys produced by ``Structure3diTransform``
        (``3di_states``) followed by ``StructureBackboneTransform`` (``mask``,
        ``indices``, ``coords_res``). Pad slots get the sentinel
        ``num_3di_classes`` ID so the embedding lookup never sees garbage
        class IDs at zero-padded positions.
        """
        states = batch["3di_states"].long()
        seq_mask = batch["mask"].float()
        residue_index = batch["indices"].long()

        states = torch.where(seq_mask.bool(), states, torch.full_like(states, self.num_3di_classes))
        return states, seq_mask, residue_index

    # ------------------------------------------------------------------
    # Forward (inference): 3Di tokens -> (B, L, 3, 3) coords
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        states, seq_mask, residue_index = self.featurize(batch)
        decoder_name = next(iter(self.decoder_factory.list_decoders()))
        out = self.decoder_factory.decoders[decoder_name](states, seq_mask, residue_index=residue_index)
        if isinstance(out, dict) and "protein_coords" in out:
            return out["protein_coords"]
        return out

    def encode_3di(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference helper: 3Di tokens (and optional positional index) → coords.

        Parameters
        ----------
        states
            ``(B, L)`` long tensor of 3Di class IDs (``[0, num_3di_classes)``).
        mask
            ``(B, L)`` float tensor (``1.0`` = valid residue, ``0.0`` = pad).
        residue_index
            Optional ``(B, L)`` long tensor; if omitted, ``arange(L)`` per row.

        Returns
        -------
        coords
            ``(B, L, 3, 3)`` predicted N/CA/C backbone coordinates.
        """
        if residue_index is None:
            B, L = states.shape
            residue_index = torch.arange(L, device=states.device)[None].repeat(B, 1)
        batch = {"3di_states": states, "mask": mask, "indices": residue_index}
        return self.forward(batch)

    # ------------------------------------------------------------------
    # Train / val
    # ------------------------------------------------------------------

    def _single_step(self, batch: dict, split: str) -> dict:
        states, seq_mask, residue_index = self.featurize(batch)

        loss_dict: dict[str, torch.Tensor] = {}
        total_loss = torch.zeros((), device=states.device)
        # `x_recon` mirrors the layout produced by `TokenizerMulti.training_step`
        # so callbacks like `BackboneReconstruction` (which expect
        # `outputs["x_recon"][decoder_name]`) continue to work unchanged.
        x_recon: dict[str, torch.Tensor] = {}

        for decoder_name in self.decoder_factory.list_decoders():
            recon = self.decoder_factory.decoders[decoder_name](states, seq_mask, residue_index=residue_index)
            x_recon[decoder_name] = recon
            losses_for_this = self.decoder_factory.get_loss(decoder_name)
            if not isinstance(losses_for_this, (list, omegaconf.ListConfig)):
                losses_for_this = [losses_for_this]
            for loss_name in losses_for_this:
                loss_val = self.loss_factory(loss_name, batch, recon, seq_mask)
                weighted = self.loss_factory.weight_dict.get(loss_name, 1.0) * loss_val
                total_loss = total_loss + weighted
                loss_dict[f"{split}_{loss_name}"] = loss_val

        B = batch["mask"].shape[0]
        self.log_dict(
            {f"{split}_loss": total_loss, **loss_dict},
            batch_size=B,
            sync_dist=True,
        )

        if self.automatic_optimization is False:
            self.manual_backward(total_loss)
            self.trainer.optimizers[0].step()

        return {"loss": total_loss, "x_recon": x_recon}

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self._single_step(batch, split="train")

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self._single_step(batch, split="val")

    def configure_optimizers(self):
        optimizer = self.optim_factory(params=self.parameters())
        scheduler = self.lr_scheduler(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
