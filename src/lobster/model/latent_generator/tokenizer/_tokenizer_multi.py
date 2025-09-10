# lightning module training script
import logging

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import math
import os
from collections.abc import Callable
from typing import Literal

import lightning.pytorch as pl
import omegaconf
import torch
from einops import rearrange

from lobster.model.latent_generator.structure_decoder import DecoderFactory
from lobster.model.latent_generator.structure_encoder import BaseEncoder
from lobster.model.latent_generator.utils.residue_constants import restype_order_with_x

from ._loss_factory import LossFactory

logger = logging.getLogger(__name__)


class TokenizerMulti(pl.LightningModule):
    """Base class for PyTorch Lightning training modules.

    This class is a subclass of the PyTorch Lightning Module class.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.

    optimizer : torch.optim.Optimizer
        The optimizer to use for training.

    criterion : Callable
        The loss function to use for training.

    """

    def __init__(
        self,
        structure_encoder: Callable[..., BaseEncoder],
        decoder_factory: Callable[..., DecoderFactory],
        loss_factory: Callable[..., LossFactory],
        optim: Callable[..., torch.optim.Optimizer],
        lr_scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        quantizer: torch.nn.Module = None,
        structure_path: str = None,
        automatic_optimization: bool = True,
        freeze_decoder: bool = False,
        freeze_encoder: bool = False,
        freeze_quantizer: bool = False,
        schedule: Literal["arccos", "cubic", "exp"] = "arccos",
        mask_input_tokens: bool = False,
        mask_coords: bool = False,
        min_mask_timestep: float = 0.5,
        mask_sequence: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        self.encoder = structure_encoder
        self.quantizer = quantizer
        self.decoder_factory = decoder_factory
        self.loss_factory = loss_factory
        self.optim_factory = optim
        self.lr_scheduler = lr_scheduler
        self.freeze_decoder = freeze_decoder
        self.freeze_encoder = freeze_encoder
        self.freeze_quantizer = freeze_quantizer
        self.debug = debug
        self.schedule = schedule
        self.mask_input_tokens = mask_input_tokens
        self.mask_coords = mask_coords
        self.min_mask_timestep = min_mask_timestep
        self.mask_sequence = mask_sequence
        logger.info(f"Using mask_input_tokens: {self.mask_input_tokens}")
        logger.info(f"Using mask_coords: {self.mask_coords}")
        logger.info(f"Using min_mask_timestep: {self.min_mask_timestep}")
        logger.info(f"Using schedule: {self.schedule}")

        self.structure_path = structure_path
        if not os.path.exists(f"{self.structure_path}train/") and self.structure_path is not None:
            os.makedirs(f"{self.structure_path}train/")

        self.automatic_optimization = automatic_optimization

    def on_after_backward(self):
        if self.debug:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print(name)

    def forward(self, x, return_embeddings=False, *args, **kwargs):
        """Forward pass of the model, for inference."""
        x_feat = self.encoder.featurize(x, *args, **kwargs)
        # check if x_feat is a tuple of length 7 or 8 for ligand case
        if len(x_feat) == 8:
            x_emb = self.encoder(
                coords=x_feat[0],
                seq_mask=x_feat[1],
                residue_index=x_feat[2],
                sequence=x_feat[3],
                ligand_coords=x_feat[4],
                ligand_mask=x_feat[5],
                ligand_residue_index=x_feat[6],
                ligand_atom_types=x_feat[7],
                return_embeddings=return_embeddings,
            )
            seq_mask = x_feat[1]
            ligand_mask = x_feat[5]
        elif len(x_feat) == 7:
            x_emb = self.encoder(
                coords=x_feat[0],
                seq_mask=x_feat[1],
                residue_index=x_feat[2],
                sequence=x_feat[3],
                ligand_coords=x_feat[4],
                ligand_mask=x_feat[5],
                ligand_residue_index=x_feat[6],
                return_embeddings=return_embeddings,
            )
            seq_mask = x_feat[1]
            ligand_mask = x_feat[5]
        else:
            x_emb = self.encoder(*x_feat, return_embeddings=return_embeddings)
            seq_mask = x_feat[1]
            ligand_mask = None
        if return_embeddings:
            x_emb, x_emb_out = x_emb

        if self.quantizer is not None:
            x_quant, x_quant_emb, mask = self.quantizer.quantize(x_emb, mask=seq_mask, ligand_mask=ligand_mask)
        else:
            x_quant = x_emb
            if ligand_mask is not None:
                if seq_mask is not None:
                    B, L = seq_mask.shape
                    z_protein = x_quant[:, :L, :]
                    z_ligand = x_quant[:, L:, :]
                    x_quant = {"ligand_tokens": z_ligand, "protein_tokens": z_protein}
                else:
                    x_quant = {"ligand_tokens": x_quant}

        if return_embeddings:
            return x_quant, x_emb_out

        return x_quant

    def decode(self, x_quant, x_emb=None, mask=None):
        """decode a sample."""
        if isinstance(x_quant, dict):
            if "protein_tokens" in x_quant:
                B, L = x_quant["protein_tokens"].shape[:2]
            B_ligand, L_ligand = x_quant["ligand_tokens"].shape[:2]
            device = x_quant["ligand_tokens"].device
            if mask is None:
                if "protein_tokens" in x_quant:
                    mask = {
                        "protein_mask": torch.ones(B, L, device=device),
                        "ligand_mask": torch.ones(B_ligand, L_ligand, device=device),
                    }
                else:
                    mask = {"ligand_mask": torch.ones(B_ligand, L_ligand, device=device)}
        else:
            B, L = x_quant.shape[:2]
            device = x_quant.device
            if mask is None:
                mask = torch.ones(B, L, device=device)

        # iterate over decoders
        x_recon = {}
        for decoder_name in self.decoder_factory.list_decoders():
            x_recon[decoder_name] = self.decoder_factory.decoders[decoder_name](x_quant, mask, x_emb=x_emb)

        return x_recon

    def generate_masks(self, batch_size: int, seq_len: int, device: torch.device, timesteps=None):
        """Generate masks for input tokens or coordinates.

        This function implements a masking strategy for training with masked tokens. It works as follows:
        1. Generates or uses provided timesteps to determine the masking ratio
        2. Calculates the number of tokens to mask based on the selected schedule (arccos/cubic/exp)
        3. Creates a random binary mask where True indicates positions to be masked

        The masking schedule options are:
        - "arccos": mask_ratio = arccos(t) / (π/2), provides smooth transition
        - "cubic": mask_ratio = 1 - t³, faster initial masking
        - "exp": mask_ratio = 1 - exp(-6(1-t)), exponential decay

        Note on timesteps:
        - t = 0: Full masking (mask_ratio = 1)
        - t = 1: No masking (mask_ratio = 0)
        - Intermediate values provide partial masking

        Args:
            batch_size: Number of samples in the batch
            seq_len: Length of the sequence
            device: Device to create tensors on
            timesteps: Optional timesteps tensor of shape (B,) in range [0,1]
                If None, random timesteps are generated
                0 means full masking, 1 means no masking

        Returns:
            tuple containing:
                masks: Boolean tensor of shape (B,L)
                    True indicates masked positions
                timesteps: Tensor of shape (B,)
                    The timesteps used for masking
        """
        # Generate timesteps if not provided
        if timesteps is None:
            timesteps = torch.zeros((batch_size,), device=device).float().uniform_(self.min_mask_timestep, 1)

        # Calculate mask ratio based on schedule
        if self.schedule == "arccos":
            mask_ratio = torch.acos(timesteps) / (math.pi * 0.5)
        elif self.schedule == "cubic":
            mask_ratio = 1.0 - timesteps**3.0
        elif self.schedule == "exp":
            mask_ratio = 1.0 - torch.exp(-6 * (1.0 - timesteps))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        # Calculate number of tokens to mask
        mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.0)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)

        # Generate random mask
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, "b -> b 1")

        return masks, timesteps

    def single_step(self, batch, batch_idx, split="train"):
        """Training step of the model."""
        if self.freeze_decoder:
            logger.info("fixing decoder weights")
            for name, param in self.named_parameters():
                if "decoder" in name:
                    if "embed_struc_tokens" in name:
                        continue
                    param.requires_grad = False

        # fix weights of encoder if needed
        if self.freeze_encoder:
            for name, param in self.named_parameters():
                name_ = name.split(".")[0]
                if "encoder" in name_ or "quantizer" in name_:
                    param.requires_grad = False

        x = batch
        if "sequence" in batch:
            B = batch["sequence"].shape[0]
            device = batch["sequence"].device
            mask = batch["mask"]
        else:
            B = batch["ligand_coords"].shape[0]
            device = batch["ligand_coords"].device
            mask = None
        if "ligand_mask" in batch:
            ligand_mask = batch["ligand_mask"]
        else:
            ligand_mask = None

        # Generate timesteps and masks if needed
        timesteps = None
        gen_masks = None
        if self.mask_input_tokens or self.mask_coords or self.mask_sequence:
            gen_masks, timesteps = self.generate_masks(B, batch["sequence"].shape[1], device)

        with torch.no_grad():
            x_feat = list(self.encoder.featurize(x))  # Convert tuple to list

        # Mask coordinates if enabled for MAE like training
        if self.mask_coords and gen_masks is not None:
            # x_feat[0] is coordinates, x_feat[1] is mask
            x_feat[0] = torch.where(gen_masks.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(x_feat[0]), x_feat[0])
            x_feat[1] = torch.where(gen_masks, torch.zeros_like(x_feat[1]), x_feat[1])

        if self.mask_sequence and gen_masks is not None:
            x_feat[3] = torch.where(gen_masks, torch.full_like(x_feat[3], restype_order_with_x["X"]), x_feat[3])

        # Handle both standard case (4 values) and ligand case (7 or 8 values)
        if len(x_feat) == 8:
            x_emb = self.encoder(
                coords=x_feat[0],
                seq_mask=x_feat[1],
                residue_index=x_feat[2],
                sequence=x_feat[3],
                ligand_coords=x_feat[4],
                ligand_mask=x_feat[5],
                ligand_residue_index=x_feat[6],
                ligand_atom_types=x_feat[7],
            )
        elif len(x_feat) == 7:
            x_emb = self.encoder(
                coords=x_feat[0],
                seq_mask=x_feat[1],
                residue_index=x_feat[2],
                sequence=x_feat[3],
                ligand_coords=x_feat[4],
                ligand_mask=x_feat[5],
                ligand_residue_index=x_feat[6],
            )
        else:
            x_emb = self.encoder(*x_feat)  # Keep original unpacking for backward compatibility

        if self.quantizer is not None:
            # check if cls token is used
            if self.encoder.add_cls_token:
                x_quant, x_quant_emb, mask = self.quantizer.quantize(
                    x_emb[:, 1:, :], mask=mask, ligand_mask=ligand_mask
                )
            else:
                x_quant, x_quant_emb, mask = self.quantizer.quantize(x_emb, mask=mask, ligand_mask=ligand_mask)
            # Apply masking to tokens if enabled
            if self.mask_input_tokens and gen_masks is not None:
                if isinstance(x_quant, dict):
                    zero_logits = torch.zeros_like(x_quant["protein_tokens"])
                    masked_protein_tokens = torch.where(gen_masks.unsqueeze(-1), zero_logits, x_quant["protein_tokens"])
                    masked_tokens = {"protein_tokens": masked_protein_tokens, "ligand_tokens": x_quant["ligand_tokens"]}
                else:
                    zero_logits = torch.zeros_like(x_quant)
                    masked_tokens = torch.where(gen_masks.unsqueeze(-1), zero_logits, x_quant)
            else:
                masked_tokens = x_quant
        else:
            x_quant = x_emb
            masked_tokens = x_quant
            if ligand_mask is not None:
                if mask is not None:
                    B, L = mask.shape
                    z_protein = x_quant[:, :L, :]
                    z_ligand = x_quant[:, L:, :]
                    masked_tokens = {"ligand_tokens": z_ligand, "protein_tokens": z_protein}
                    mask = {"ligand_mask": ligand_mask, "protein_mask": mask}
                else:
                    masked_tokens = {"ligand_tokens": x_quant}
                    mask = {"ligand_mask": ligand_mask}

        # iterate over decoders
        x_recon = {}
        loss_dict = {}
        total_loss = 0
        for i, decoder_name in enumerate(self.decoder_factory.list_decoders()):
            x_recon[decoder_name] = self.decoder_factory.decoders[decoder_name](
                masked_tokens, mask, x_emb=x_emb, batch=batch, cls_token=self.encoder.add_cls_token
            )

            # apply loss
            loss2apply = self.decoder_factory.get_loss(decoder_name)

            # check if list of losses
            if not isinstance(loss2apply, omegaconf.ListConfig):
                loss2apply = [loss2apply]

            for loss2apply_ in loss2apply:
                loss = self.loss_factory(loss2apply_, batch, x_recon[decoder_name], mask)
                # apply loss weighting from weight_dict in loss_factory
                total_loss += self.loss_factory.weight_dict[loss2apply_] * loss
                loss_dict[f"{split}_{loss2apply_}"] = loss

        if timesteps is not None:
            self.log_dict(
                {f"{split}_loss": total_loss, f"{split}_t": timesteps.mean(), **loss_dict}, batch_size=B, sync_dist=True
            )
        else:
            self.log_dict({f"{split}_loss": total_loss, **loss_dict}, batch_size=B, sync_dist=True)

        if self.automatic_optimization is False:
            self.manual_backward(total_loss)
            # Manually update parameters
            self.trainer.optimizers[0].step()

        return {"loss": total_loss, "x_recon": x_recon}

    def training_step(self, batch, batch_idx):
        """Training step of the model."""
        return self.single_step(batch, batch_idx, split="train")

    def validation_step(self, batch, batch_idx):  # mask, t):
        """Validation step of the model."""
        return self.single_step(batch, batch_idx, split="val")

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = self.optim_factory(params=self.parameters())

        out = {"optimizer": optimizer}

        out["lr_scheduler"] = {"scheduler": self.lr_scheduler(optimizer=optimizer), "interval": "step"}

        return out

    def predict_step(self, batch, batch_idx):
        """prediction step of the model."""
        x = batch
        B = batch["sequence"].shape[0]
        mask = batch["mask"]

        with torch.no_grad():
            x_feat = self.encoder.featurize(x)
        x_emb = self.encoder(*x_feat)

        if self.quantizer is not None:
            x_quant, x_quant_emb, mask = self.quantizer.quantize(x_emb, mask=mask, batch_size=B)
        else:
            x_quant = x_emb

        # iterate over decoders
        x_recon = {}
        for i, decoder_name in enumerate(self.decoder_factory.list_decoders()):
            x_recon[decoder_name] = self.decoder_factory.decoders[decoder_name](
                x_quant[:B],
                mask[:B],
            )

        return {
            "x_recon": x_recon,
            "x_input": x_feat,
            "x_quant": x_quant[:B],
            "input_sequence": x["sequence"],
            "description": x["description"],
        }
