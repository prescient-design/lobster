"""LeFlur Protein-Ligand Lightning Module.

This module provides the PyTorch Lightning training wrapper for the
LeFlur protein-ligand encoder. It handles:
- Multi-modal loss computation (protein_seq, protein_struct, ligand_atom, ligand_struct, bond)
- Mixed batch handling with validity masks
- Independent time sampling for different modalities
- Structure encoding/decoding via LatentGenerator

Key features:
- Backward compatible with protein-only training
- Supports mixed protein-only + protein-ligand batches
- Flow matching for all modalities
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from lobster.model.losses._diffusion_loss import DiffusionLoss

import torch
import torch.nn as nn
import transformers
from lightning import LightningModule
from torch import Tensor
from tqdm import tqdm

from lobster.model.latent_generator.cmdline import LatentEncoderDecoder
from lobster.model.latent_generator.cmdline import methods as latent_generator_methods
from lobster.model.latent_generator.utils import apply_se3_augmentation_protein_ligand
from lobster.model.latent_generator.utils.residue_constants import NUM_BOND_TYPES

from ._bond_prediction import BondMatrixLoss
from ._leflur_protein_ligand_encoder import AuxiliaryTask, LeFlurProteinLigandEncoderModule

# Re-route the LeFlur-paired LG codecs to their HuggingFace mirror so
# external users (without internal s3:// or /cv/ access) can sample.
from .checkpoints import install_paired_lg_codec_overrides

install_paired_lg_codec_overrides()

# Imported after install_paired_lg_codec_overrides() so the LG codec path
# patches are applied before any LG-consuming code runs.
from lobster.model.neobert._config import NEOBERT_CONFIGS  # noqa: E402

# Bionemo interpolant code
from bionemo.moco.distributions.prior import DiscreteMaskedPrior, DiscreteUniformPrior  # noqa: E402
from bionemo.moco.distributions.time import UniformTimeDistribution  # noqa: E402
from bionemo.moco.interpolants import DiscreteFlowMatcher  # noqa: E402
from bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule, LogInferenceSchedule  # noqa: E402

logger = logging.getLogger(__name__)


class LeFlurProteinLigandLightningModule(LightningModule):
    """PyTorch Lightning module for LeFlur protein-ligand training.

    This module extends the protein-only LeFlur encoder to support both
    protein-only
    and protein-ligand tasks. When no ligand data is present in a batch,
    it behaves identically to the protein-only LeFlur encoder.

    Parameters
    ----------
    mask_token_id : int
        Mask token ID for protein sequences.
    pad_token_id : int
        Padding token ID for protein sequences.
    vocab_size : int
        Vocabulary size for protein sequences.
    ligand_atom_vocab_size : int
        Vocabulary size for ligand atom types (default: 25).
    ligand_mask_token_id : int
        Mask token ID for ligand atoms (default: 1 for MASK).
    ligand_pad_token_id : int
        Padding token ID for ligand atoms (default: 0 for PAD).
    auxiliary_tasks : list[AuxiliaryTask], optional
        List of auxiliary task configurations.
    seed : int
        Random seed.
    lr : float
        Learning rate.
    encoder_kwargs : dict, optional
        Additional kwargs for the encoder.
    latent_generator_model_name : str
        Name of the LatentGenerator model for structure encoding/decoding.
    use_masked_prior : bool
        Whether to use masked prior for flow matching.
    inverse_folding : bool
        Whether to train for inverse folding (structure -> sequence).
    bond_loss_weight : float
        Weight for bond prediction loss.
    """

    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        ligand_atom_vocab_size: int = 25,
        ligand_mask_token_id: int = 1,
        ligand_pad_token_id: int = 0,
        num_bond_types: int = NUM_BOND_TYPES,
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        seed: int = 0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_warmup_steps: int = 20_000,
        num_training_steps: int = 100_000,
        weight_decay: float = 0.0,
        scheduler: str = "constant",
        scheduler_kwargs: dict | None = None,
        encoder_kwargs: dict | None = None,
        ckpt_path: str | None = None,
        # LatentGenerator params
        decode_tokens_during_training: bool = True,
        latent_generator_model_name: str = "LG Protein Ligand fsq 4375",
        # Generation params
        prior_distribution_seq: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        prior_distribution_struc: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        prior_distribution_ligand_atom: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        prior_distribution_ligand_struc: Callable[..., DiscreteUniformPrior] = DiscreteUniformPrior,
        time_distribution_seq: Callable[..., UniformTimeDistribution] = UniformTimeDistribution,
        time_distribution_struc: Callable[..., UniformTimeDistribution] = UniformTimeDistribution,
        time_distribution_ligand: Callable[..., UniformTimeDistribution] = UniformTimeDistribution,
        interpolant: Callable[..., DiscreteFlowMatcher] = DiscreteFlowMatcher,
        inference_schedule: Callable[..., LinearInferenceSchedule] = LinearInferenceSchedule,
        use_masked_prior: bool = True,
        inverse_folding: bool = False,
        # Loss weights
        bond_loss_weight: float = 1.0,
        ligand_atom_loss_weight: float = 1.0,
        ligand_struct_loss_weight: float = 1.0,
        # Diffusion loss params for continuous structure tokens
        use_diffusion_loss_structure: bool = False,
        diffusion_target_dim: int = 256,  # Structure embedding dim from LatentGenerator
        diffusion_z_dim: int | None = None,  # Transformer hidden dim (auto-detect if None)
        diffusion_depth: int = 3,  # MLP depth (from MAR: diffloss_d)
        diffusion_width: int = 1024,  # MLP width (from MAR: diffloss_w)
        diffusion_num_sampling_steps: str = "100",
        diffusion_noise_schedule: Literal["linear", "cosine"] = "cosine",
        diffusion_loss_weight: float = 1.0,
        # SE(3) augmentation
        use_se3_augmentation: bool = True,
        se3_translation_scale: float = 1.0,
    ):
        self.save_hyperparameters()
        super().__init__()

        # Store config
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.ligand_atom_vocab_size = ligand_atom_vocab_size
        self.ligand_mask_token_id = ligand_mask_token_id
        self.ligand_pad_token_id = ligand_pad_token_id
        self.num_bond_types = num_bond_types

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.seed = seed

        # Loss weights
        self.bond_loss_weight = bond_loss_weight
        self.ligand_atom_loss_weight = ligand_atom_loss_weight
        self.ligand_struct_loss_weight = ligand_struct_loss_weight

        # SE(3) augmentation config
        self.use_se3_augmentation = use_se3_augmentation
        self.se3_translation_scale = se3_translation_scale

        # Auxiliary tasks
        self.auxiliary_tasks = auxiliary_tasks
        self.auxiliary_task_loss_fns = {"regression": nn.MSELoss()}

        # LatentGenerator for structure encoding/decoding
        self.decode_tokens_during_training = decode_tokens_during_training
        self.structure_latent_encoder_decoder = LatentEncoderDecoder()

        # Load from registered model name
        logger.info(f"Loading LatentGenerator model: {latent_generator_model_name}")
        self.structure_latent_encoder_decoder.load_model(
            latent_generator_methods[latent_generator_model_name].model_config.checkpoint,
            latent_generator_methods[latent_generator_model_name].model_config.config_path,
            latent_generator_methods[latent_generator_model_name].model_config.config_name,
            overrides=latent_generator_methods[latent_generator_model_name].model_config.overrides,
        )
        self.quantizer = self.structure_latent_encoder_decoder.model.quantizer
        self.structure_encoder = self.structure_latent_encoder_decoder.model.encoder
        self.decoder_factory = self.structure_latent_encoder_decoder.model.decoder_factory
        self.loss_factory = self.structure_latent_encoder_decoder.model.loss_factory

        # === CONTINUOUS/DISCRETE MODE SETUP ===
        # Check if using continuous mode (quantizer is None)
        self.use_continuous_structure = self.quantizer is None
        if self.use_continuous_structure:
            # For continuous mode, we need a way to handle masking
            # Use a simple vocab: 0 = visible (placeholder), 1 = mask, 2 = pad
            # The actual embedding values come from continuous embeddings, not discrete tokens
            self.structure_embed_dim = self.structure_encoder.embed_dim
            self.num_struc_classes = 3  # placeholder, mask, pad
            self.mask_index_struc_tokens = 1
            self.padding_index_struc_tokens = 2
            logger.info(f"Using CONTINUOUS structure embeddings (dim={self.structure_embed_dim})")
        else:
            self.structure_embed_dim = None
            self.num_struc_classes = self.quantizer.n_tokens + 2
            self.mask_index_struc_tokens = self.quantizer.n_tokens
            self.padding_index_struc_tokens = self.quantizer.n_tokens + 1
            logger.info(f"Using DISCRETE structure tokens (vocab={self.quantizer.n_tokens})")

        # === FLOW MATCHING SETUP ===
        self.inverse_folding = inverse_folding

        # Set up priors and interpolants
        device = "cpu"  # Will be moved to correct device during training

        # Protein sequence interpolant
        if use_masked_prior:
            prior_seq = DiscreteMaskedPrior(num_classes=self.vocab_size, mask_dim=self.mask_token_id, inclusive=True)
            prior_struc = DiscreteMaskedPrior(
                num_classes=self.num_struc_classes, mask_dim=self.mask_index_struc_tokens, inclusive=True
            )
            prior_ligand_atom = DiscreteMaskedPrior(
                num_classes=self.ligand_atom_vocab_size, mask_dim=self.ligand_mask_token_id, inclusive=True
            )
            prior_ligand_struc = DiscreteMaskedPrior(
                num_classes=self.num_struc_classes, mask_dim=self.mask_index_struc_tokens, inclusive=True
            )
        else:
            prior_seq = prior_distribution_seq(num_classes=self.vocab_size)
            prior_struc = prior_distribution_struc(num_classes=self.num_struc_classes)
            prior_ligand_atom = prior_distribution_ligand_atom(num_classes=self.ligand_atom_vocab_size)
            prior_ligand_struc = prior_distribution_ligand_struc(num_classes=self.num_struc_classes)

        time_dist_seq = time_distribution_seq()
        time_dist_struc = time_distribution_struc()
        time_dist_ligand = time_distribution_ligand()

        # Create interpolants
        self.interpolant_seq = interpolant(time_distribution=time_dist_seq, prior_distribution=prior_seq, device=device)
        self.interpolant_struc = interpolant(
            time_distribution=time_dist_struc, prior_distribution=prior_struc, device=device
        )
        self.interpolant_ligand_atom = interpolant(
            time_distribution=time_dist_ligand, prior_distribution=prior_ligand_atom, device=device
        )
        self.interpolant_ligand_struc = interpolant(
            time_distribution=time_dist_ligand, prior_distribution=prior_ligand_struc, device=device
        )

        self.inference_schedule = inference_schedule(nsteps=1000)

        logger.info(f"Initialized protein-ligand flow matching with masked_prior={use_masked_prior}")

        # === ENCODER ===
        self.encoder = LeFlurProteinLigandEncoderModule(
            auxiliary_tasks=auxiliary_tasks,
            sequence_token_vocab_size=self.vocab_size,
            structure_token_vocab_size=self.num_struc_classes,
            ligand_atom_vocab_size=self.ligand_atom_vocab_size,
            ligand_structure_vocab_size=self.num_struc_classes,
            sequence_token_pad_token_id=self.pad_token_id,
            structure_token_pad_token_id=self.padding_index_struc_tokens,
            ligand_atom_pad_token_id=self.ligand_pad_token_id,
            ligand_structure_pad_token_id=self.padding_index_struc_tokens,
            num_bond_types=self.num_bond_types,
            model_ckpt=ckpt_path,
            **encoder_kwargs or {},
        )

        # === BOND LOSS ===
        self.bond_loss_fn = BondMatrixLoss(ignore_diagonal=True)

        # === DIFFUSION LOSS FOR STRUCTURE (Option A: Hybrid) ===
        self.use_diffusion_loss_structure = use_diffusion_loss_structure
        self.diffusion_loss_weight = diffusion_loss_weight
        self.diffusion_loss_protein_struc: DiffusionLoss | None = None
        self.diffusion_loss_ligand_struc: DiffusionLoss | None = None

        if use_diffusion_loss_structure:
            try:
                from lobster.model.losses._diffusion_loss import DiffusionLoss
            except ImportError as exc:
                raise ImportError(
                    "use_diffusion_loss_structure=True requires the optional "
                    "lobster.model.losses._diffusion_loss module, which is not "
                    "bundled with the publication wheel (research-only feature). "
                    "No canonical LeFlur checkpoint (leflur-base, leflur-ted, "
                    "leflur-pl) sets this flag — leave it at the default False, "
                    "or obtain _diffusion_loss.py from the internal repository "
                    "to re-enable the continuous-structure variant."
                ) from exc

            # Auto-detect transformer hidden dim if not provided
            if diffusion_z_dim is not None:
                z_dim = diffusion_z_dim
            elif encoder_kwargs.get("model_size") in NEOBERT_CONFIGS:
                z_dim = NEOBERT_CONFIGS[encoder_kwargs["model_size"]]["hidden_size"]
            else:
                z_dim = 768  # Default fallback

            # Initialize DiffusionLoss for protein structure
            self.diffusion_loss_protein_struc = DiffusionLoss(
                target_channels=diffusion_target_dim,
                z_channels=z_dim,
                depth=diffusion_depth,
                width=diffusion_width,
                num_sampling_steps=diffusion_num_sampling_steps,
                noise_schedule=diffusion_noise_schedule,
            )

            # Initialize DiffusionLoss for ligand structure
            self.diffusion_loss_ligand_struc = DiffusionLoss(
                target_channels=diffusion_target_dim,
                z_channels=z_dim,
                depth=diffusion_depth,
                width=diffusion_width,
                num_sampling_steps=diffusion_num_sampling_steps,
                noise_schedule=diffusion_noise_schedule,
            )

            # Initialize continuous structure projection in encoder
            # This allows feeding back sampled embeddings during MAR-style generation
            self.encoder.init_continuous_structure_proj(diffusion_target_dim)

            logger.info(
                f"Using DiffusionLoss for structure tokens "
                f"(target_dim={diffusion_target_dim}, z_dim={z_dim}, "
                f"depth={diffusion_depth}, width={diffusion_width})"
            )

    def encode_structure(
        self, x_gt: Tensor, mask: Tensor, residue_index: Tensor, return_continuous: bool = False
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode protein structure to tokens using LatentGenerator.

        Handles both FSQLigandTokenizer (returns dicts), standard quantizers,
        and continuous mode (quantizer=None).

        Parameters
        ----------
        x_gt : Tensor
            Ground truth coordinates [B, L, 4*3].
        mask : Tensor
            Residue mask [B, L].
        residue_index : Tensor
            Residue indices [B, L].
        return_continuous : bool
            If True, also return continuous embeddings for DiffusionLoss.

        Returns
        -------
        x_quant : Tensor
            Soft token distribution with shape [B, L, n_tokens] for FSQLigandTokenizer,
            quantized logits for standard quantizers, or dummy tokens for continuous mode.
        x_quant_emb : Tensor
            Embeddings from encoder.
        out_mask : Tensor
            Mask for valid positions.
        x_continuous : Tensor (only if return_continuous=True)
            Raw continuous embeddings [B, L, embed_dim].
        """
        x_emb = self.structure_encoder(x_gt, mask, residue_index=residue_index)

        if self.use_continuous_structure:
            # Continuous mode: no quantization
            # Return dummy discrete tokens (all zeros, will be masked anyway)
            # The continuous embeddings are what we actually use
            B, L, D = x_emb.shape
            x_quant = torch.zeros(B, L, self.num_struc_classes, device=x_emb.device)
            x_quant[:, :, 0] = 1.0  # All "visible" placeholder token
            x_quant_emb = x_emb
            out_mask = mask

            if return_continuous:
                return x_quant, x_quant_emb, out_mask, x_emb
            return x_quant, x_quant_emb, out_mask

        # Handle FSQLigandTokenizer which returns dicts
        result = self.quantizer.quantize(x_emb, mask=mask)
        if isinstance(result[0], dict):
            # FSQLigandTokenizer returns (tokens_dict, logits_dict, masks_dict)
            # tokens_dict contains soft token distributions [B, L, n_tokens]
            # logits_dict contains raw FSQ logits [B, L, 5] (FSQ levels)
            tokens_dict, logits_dict, masks_dict = result
            # Use tokens (vocabulary size) not logits (FSQ levels) for proper token extraction
            x_quant = tokens_dict.get("protein_tokens", tokens_dict.get("ligand_tokens"))
            x_quant_emb = x_emb  # Use embeddings as-is
            out_mask = masks_dict.get("protein_mask", masks_dict.get("ligand_mask"))
        else:
            # Standard quantizer returns (x_quant, x_quant_emb, mask)
            x_quant, x_quant_emb, out_mask = result

        if return_continuous:
            return x_quant, x_quant_emb, out_mask, x_emb
        return x_quant, x_quant_emb, out_mask

    def encode_ligand_structure(
        self, ligand_coords: Tensor, ligand_mask: Tensor, atom_indices: Tensor, return_continuous: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Encode ligand structure to tokens using LatentGenerator.

        Note: The ViT encoder accepts ligand_coords as a separate argument with shape [B, N_atoms, 3].

        Parameters
        ----------
        ligand_coords : Tensor
            Ligand coordinates [B, N_atoms, 3].
        ligand_mask : Tensor
            Ligand atom mask [B, N_atoms].
        atom_indices : Tensor
            Atom indices [B, N_atoms].
        return_continuous : bool
            If True, also return continuous embeddings for DiffusionLoss.

        Returns
        -------
        x_quant_argmax : Tensor
            Structure token indices with shape [B, N_atoms].
        out_mask : Tensor
            Mask for valid positions.
        x_continuous : Tensor (only if return_continuous=True)
            Raw continuous embeddings [B, N_atoms, embed_dim].
        """
        # Ligand coords: [B, N_atoms, 3] - passed directly to encoder
        # The encoder handles ligands separately via ligand_coords argument
        x_emb = self.structure_encoder(
            coords=None,  # No protein coords
            seq_mask=torch.zeros(ligand_coords.shape[0], 1, device=ligand_coords.device),  # Dummy protein mask
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            ligand_residue_index=atom_indices,
        )

        if self.use_continuous_structure:
            # Continuous mode: no quantization
            # Return dummy discrete tokens (all masked)
            B, N = ligand_coords.shape[:2]
            x_quant_argmax = torch.full((B, N), self.mask_index_struc_tokens, device=ligand_coords.device)
            x_quant_argmax[~ligand_mask.bool()] = self.padding_index_struc_tokens
            out_mask = ligand_mask

            if return_continuous:
                return x_quant_argmax, out_mask, x_emb
            return x_quant_argmax, out_mask

        # Handle FSQLigandTokenizer which returns dicts
        result = self.quantizer.quantize(x_emb, mask=None, ligand_mask=ligand_mask)
        if isinstance(result[0], dict):
            # FSQLigandTokenizer returns (tokens_dict, logits_dict, masks_dict)
            # tokens_dict contains soft token distributions [B, L, n_tokens]
            # logits_dict contains raw FSQ logits [B, L, 5] (FSQ levels)
            tokens_dict, logits_dict, masks_dict = result
            # Use tokens (vocabulary size) not logits (FSQ levels) for proper token extraction
            x_quant = tokens_dict.get("ligand_tokens")
            out_mask = masks_dict.get("ligand_mask")
        else:
            # Standard quantizer
            x_quant, _, out_mask = result

        x_quant_argmax = torch.argmax(x_quant, dim=-1)
        x_quant_argmax[~out_mask.bool()] = self.padding_index_struc_tokens

        if return_continuous:
            return x_quant_argmax, out_mask, x_emb
        return x_quant_argmax, out_mask

    def encode_protein_ligand_structure(
        self,
        protein_coords: Tensor,
        protein_mask: Tensor,
        protein_indices: Tensor,
        ligand_coords: Tensor,
        ligand_mask: Tensor,
        ligand_indices: Tensor,
        ligand_atom_types: Tensor | None = None,
        bond_matrix: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode protein and ligand structure JOINTLY for proper interaction.

        This method encodes both protein and ligand together through the structure
        encoder, allowing cross-attention between them. The embeddings are then
        split and quantized separately.

        Parameters
        ----------
        protein_coords : Tensor
            Protein coordinates [B, L, n_atoms, 3].
        protein_mask : Tensor
            Protein mask [B, L].
        protein_indices : Tensor
            Protein residue indices [B, L].
        ligand_coords : Tensor
            Ligand coordinates [B, N_atoms, 3].
        ligand_mask : Tensor
            Ligand atom mask [B, N_atoms].
        ligand_indices : Tensor
            Ligand atom indices [B, N_atoms].
        ligand_atom_types : Tensor, optional
            Ligand atom type indices [B, N_atoms].
        bond_matrix : Tensor, optional
            Bond matrix [B, N_atoms, N_atoms].

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing:
            - "protein_tokens": Protein structure tokens [B, L]
            - "protein_mask": Protein mask [B, L]
            - "protein_embeddings": Continuous protein embeddings [B, L, D]
            - "ligand_tokens": Ligand structure tokens [B, N_atoms]
            - "ligand_mask": Ligand mask [B, N_atoms]
            - "ligand_embeddings": Continuous ligand embeddings [B, N_atoms, D]
        """
        L = protein_coords.shape[1]

        # Joint encoding: encode protein and ligand together
        joint_emb = self.structure_encoder(
            coords=protein_coords,
            seq_mask=protein_mask,
            residue_index=protein_indices,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            ligand_residue_index=ligand_indices,
            ligand_atom_types=ligand_atom_types,
            ligand_bond_matrix=bond_matrix,
        )
        # joint_emb shape: [B, L + N_atoms, D]

        # Split embeddings back into protein and ligand parts
        protein_emb = joint_emb[:, :L, :]  # [B, L, D]
        ligand_emb = joint_emb[:, L:, :]  # [B, N_atoms, D]

        # Quantize protein embeddings
        if self.use_continuous_structure:
            # Continuous mode: no quantization, dummy tokens
            B, L_prot, D = protein_emb.shape
            protein_logits = torch.zeros(B, L_prot, self.num_struc_classes, device=protein_emb.device)
            protein_logits[:, :, 0] = 1.0
            protein_out_mask = protein_mask
        else:
            # Discrete mode: quantize
            result = self.quantizer.quantize(protein_emb, mask=protein_mask)
            if isinstance(result[0], dict):
                tokens_dict, _, masks_dict = result
                protein_logits = tokens_dict.get("protein_tokens", tokens_dict.get("ligand_tokens"))
                protein_out_mask = masks_dict.get("protein_mask", masks_dict.get("ligand_mask"))
            else:
                protein_logits, _, protein_out_mask = result

        protein_tokens = torch.argmax(protein_logits, dim=-1)
        protein_tokens[~protein_out_mask.bool()] = self.padding_index_struc_tokens

        # Quantize ligand embeddings
        if self.use_continuous_structure:
            # Continuous mode: dummy discrete tokens
            B, N = ligand_coords.shape[:2]
            ligand_tokens = torch.full((B, N), self.mask_index_struc_tokens, device=ligand_coords.device)
            ligand_tokens[~ligand_mask.bool()] = self.padding_index_struc_tokens
            ligand_out_mask = ligand_mask
        else:
            # Discrete mode: quantize
            result = self.quantizer.quantize(ligand_emb, mask=None, ligand_mask=ligand_mask)
            if isinstance(result[0], dict):
                tokens_dict, _, masks_dict = result
                ligand_logits = tokens_dict.get("ligand_tokens")
                ligand_out_mask = masks_dict.get("ligand_mask")
            else:
                ligand_logits, _, ligand_out_mask = result
            ligand_tokens = torch.argmax(ligand_logits, dim=-1)
            ligand_tokens[~ligand_out_mask.bool()] = self.padding_index_struc_tokens

        return {
            "protein_tokens": protein_tokens,
            "protein_mask": protein_out_mask,
            "protein_embeddings": protein_emb,
            "ligand_tokens": ligand_tokens,
            "ligand_mask": ligand_out_mask,
            "ligand_embeddings": ligand_emb,
        }

    def decode_structure(
        self,
        generated_output: dict[str, Tensor],
        mask: Tensor,
        ligand_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Decode protein and ligand structure tokens back to 3D coordinates.

        This method handles both protein-only and protein-ligand decoding.
        When ligand data is present, it passes both to the vit_decoder which
        returns a dict with "protein_coords" and "ligand_coords".

        For continuous mode (use_diffusion_loss_structure=True), pass continuous
        embeddings directly via "structure_embeddings" key.

        Parameters
        ----------
        generated_output : dict
            Output from forward pass or generate_sample, containing structure_logits
            and optionally ligand_structure_logits. For continuous mode, use
            "structure_embeddings" and "ligand_structure_embeddings" keys.
        mask : Tensor
            [B, L] mask for valid protein positions
        ligand_mask : Tensor, optional
            [B, N_atoms] mask for valid ligand positions

        Returns
        -------
        dict
            Dictionary with decoder name as key and decoded coords as value.
            When ligand is present, value is a dict with "protein_coords" and "ligand_coords".
            e.g., {"vit_decoder": {"protein_coords": Tensor[B, L, 3, 3], "ligand_coords": Tensor[B, N, 3]}}
        """
        decoder_name = "vit_decoder"
        decoded_x = {}

        # Check for continuous mode (direct embeddings instead of logits)
        if "structure_embeddings" in generated_output:
            # Continuous mode: use embeddings directly
            struc_tokens_ = generated_output["structure_embeddings"]
            has_ligand = "ligand_structure_embeddings" in generated_output and ligand_mask is not None

            if has_ligand:
                lig_struc_tokens_ = generated_output["ligand_structure_embeddings"]
                x_quant = {
                    "protein_tokens": struc_tokens_,
                    "ligand_tokens": lig_struc_tokens_,
                }
                mask_dict = {
                    "protein_mask": mask,
                    "ligand_mask": ligand_mask,
                }
                decoded_x[decoder_name] = self.decoder_factory.decoders[decoder_name](x_quant, mask_dict)
            else:
                decoded_x[decoder_name] = self.decoder_factory.decoders[decoder_name](struc_tokens_, mask)

            return decoded_x

        # Discrete mode: use logits
        # Get protein structure logits
        if "structure_logits" in generated_output:
            struc_logits = generated_output["structure_logits"]
        else:
            raise ValueError("No structure_logits or structure_embeddings found in output")

        # Handle continuous mode where quantizer may be None
        if self.use_continuous_structure or self.quantizer is None:
            # For continuous mode with logits (shouldn't normally happen, but handle gracefully)
            struc_tokens_ = struc_logits
        else:
            # Slice to only valid token indices (exclude mask/pad tokens)
            struc_tokens = struc_logits[..., : self.quantizer.n_tokens]
            # Apply softmax with temperature for soft decoding
            temp = 0.1
            struc_tokens_ = torch.softmax(struc_tokens / temp, dim=-1)

        # Check if ligand data is present
        has_ligand = "ligand_structure_logits" in generated_output and ligand_mask is not None and ligand_mask.sum() > 0

        if has_ligand:
            # Get ligand structure logits
            lig_struc_logits = generated_output["ligand_structure_logits"]

            # Handle continuous mode where quantizer may be None
            if self.use_continuous_structure or self.quantizer is None:
                lig_struc_tokens_ = lig_struc_logits
            else:
                lig_struc_tokens = lig_struc_logits[..., : self.quantizer.n_tokens]
                temp = 0.1
                lig_struc_tokens_ = torch.softmax(lig_struc_tokens / temp, dim=-1)

            # Prepare dict input for vit_decoder (matching TokenizerMulti.decode pattern)
            x_quant = {
                "protein_tokens": struc_tokens_,
                "ligand_tokens": lig_struc_tokens_,
            }
            mask_dict = {
                "protein_mask": mask,
                "ligand_mask": ligand_mask,
            }

            # Decode through vit_decoder - returns {"protein_coords": ..., "ligand_coords": ...}
            decoded_x[decoder_name] = self.decoder_factory.decoders[decoder_name](x_quant, mask_dict)
        else:
            # Protein-only decoding
            decoded_x[decoder_name] = self.decoder_factory.decoders[decoder_name](struc_tokens_, mask)

        return decoded_x

    def extract_ligand_predictions(self, generated_output: dict[str, Tensor], ligand_mask: Tensor) -> dict[str, Tensor]:
        """Extract ligand atom type and bond predictions from model output.

        Note: Coordinate decoding is done by decode_structure() which handles
        both protein and ligand together using vit_decoder. This method extracts
        the discrete predictions (atom types, bonds) from the flow matching output.

        Parameters
        ----------
        generated_output : dict
            Output containing ligand_atom_logits and bond_logits
        ligand_mask : Tensor
            [B, N_atoms] mask for valid ligand atoms

        Returns
        -------
        dict
            Dictionary with ligand predictions:
            - "coords": None (populated later from decode_structure output)
            - "atom_types": Tensor[B, N_atoms] (argmax of atom logits)
            - "bond_matrix": Tensor[B, N_atoms, N_atoms] (argmax of bond logits)
        """
        ligand_predictions = {}

        # Coordinates are decoded by decode_structure() - placeholder for caller to fill
        ligand_predictions["coords"] = None

        # Get atom types from atom logits
        if "ligand_atom_logits" in generated_output:
            ligand_predictions["atom_types"] = generated_output["ligand_atom_logits"].argmax(dim=-1)

        # Get bond matrix from bond logits
        if "bond_logits" in generated_output:
            ligand_predictions["bond_matrix"] = generated_output["bond_logits"].argmax(dim=-1)

        return ligand_predictions

    def get_timesteps(self, batch_size: int, has_ligand: bool = False) -> dict[str, Tensor]:
        """Sample timesteps for all modalities."""
        timesteps = {
            "sequence_tokens": self.interpolant_seq.sample_time(batch_size),
            "structure_tokens": self.interpolant_struc.sample_time(batch_size),
        }
        if has_ligand:
            timesteps["ligand_atom_tokens"] = self.interpolant_ligand_atom.sample_time(batch_size)
            timesteps["ligand_structure_tokens"] = self.interpolant_ligand_struc.sample_time(batch_size)
        return timesteps

    def interpolate_tokens(self, input_tokens: dict[str, Tensor], timesteps: dict[str, Tensor]) -> dict[str, Tensor]:
        """Interpolate tokens for flow matching."""
        x_t = {}

        # Protein sequence
        x_1_seq = input_tokens["sequence_tokens"]
        x_0_seq = self.interpolant_seq.sample_prior(x_1_seq.shape)
        t_seq = timesteps["sequence_tokens"]
        x_t["sequence_tokens"] = self.interpolant_seq.interpolate(x_1_seq, t_seq, x_0_seq)

        # Protein structure
        x_1_struc = input_tokens["structure_tokens"]
        x_0_struc = self.interpolant_struc.sample_prior(x_1_struc.shape)
        t_struc = timesteps["structure_tokens"]
        if self.inverse_folding:
            t_struc = torch.ones_like(t_struc)
        x_t["structure_tokens"] = self.interpolant_struc.interpolate(x_1_struc, t_struc, x_0_struc)

        # Ligand atom types (if present)
        if "ligand_atom_tokens" in input_tokens:
            x_1_lig_atom = input_tokens["ligand_atom_tokens"]
            x_0_lig_atom = self.interpolant_ligand_atom.sample_prior(x_1_lig_atom.shape)
            t_lig = timesteps["ligand_atom_tokens"]
            x_t["ligand_atom_tokens"] = self.interpolant_ligand_atom.interpolate(x_1_lig_atom, t_lig, x_0_lig_atom)

        # Ligand structure (if present)
        if "ligand_structure_tokens" in input_tokens:
            x_1_lig_struc = input_tokens["ligand_structure_tokens"]
            x_0_lig_struc = self.interpolant_ligand_struc.sample_prior(x_1_lig_struc.shape)
            t_lig_struc = timesteps["ligand_structure_tokens"]
            x_t["ligand_structure_tokens"] = self.interpolant_ligand_struc.interpolate(
                x_1_lig_struc, t_lig_struc, x_0_lig_struc
            )

        return x_t

    def forward(
        self,
        x_t: dict[str, Tensor],
        mask: Tensor,
        residue_index: Tensor,
        conditioning_tensor: Tensor,
        timesteps: dict[str, Tensor] | None = None,
        # Ligand inputs
        ligand_mask: Tensor | None = None,
        bond_matrix: Tensor | None = None,
        protein_valid_mask: Tensor | None = None,
        ligand_valid_mask: Tensor | None = None,
        # Continuous structure embeddings (for MAR-style generation)
        structure_embeddings: Tensor | None = None,
        ligand_structure_embeddings: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass through encoder.

        Parameters
        ----------
        x_t : dict[str, Tensor]
            Input tokens (sequence, structure, optionally ligand).
        mask : Tensor
            Attention mask.
        residue_index : Tensor
            Residue indices.
        conditioning_tensor : Tensor
            Conditioning input.
        timesteps : dict[str, Tensor], optional
            Timesteps for flow matching.
        ligand_mask : Tensor, optional
            Ligand mask.
        bond_matrix : Tensor, optional
            Bond matrix.
        protein_valid_mask : Tensor, optional
            Valid protein mask.
        ligand_valid_mask : Tensor, optional
            Valid ligand mask.
        structure_embeddings : Tensor, optional
            Pre-computed continuous protein structure embeddings [B, L, D].
            Used for MAR-style iterative generation where previously sampled
            embeddings are fed back into the model.
        ligand_structure_embeddings : Tensor, optional
            Pre-computed continuous ligand structure embeddings [B, N_atoms, D].
        """
        # Expand timesteps if provided
        if timesteps is not None:
            timesteps = timesteps.copy()
            B, L = x_t["sequence_tokens"].shape
            timesteps["sequence_tokens"] = timesteps["sequence_tokens"][:, None].expand(-1, L)[:, :, None]
            timesteps["structure_tokens"] = timesteps["structure_tokens"][:, None].expand(-1, L)[:, :, None]

        # Extract ligand tokens if present
        ligand_atom_input_ids = x_t.get("ligand_atom_tokens")
        ligand_structure_input_ids = x_t.get("ligand_structure_tokens")

        output = self.encoder(
            sequence_input_ids=x_t["sequence_tokens"],
            structure_input_ids=x_t["structure_tokens"],
            attention_mask=mask,
            conditioning_tensor=conditioning_tensor,
            ligand_atom_input_ids=ligand_atom_input_ids,
            ligand_structure_input_ids=ligand_structure_input_ids,
            ligand_mask=ligand_mask,
            bond_matrix=bond_matrix,
            protein_valid_mask=protein_valid_mask,
            ligand_valid_mask=ligand_valid_mask,
            timesteps=timesteps,
            structure_embeddings=structure_embeddings,
            ligand_structure_embeddings=ligand_structure_embeddings,
        )

        return output

    def compute_loss(
        self,
        split: str,
        x_gt: dict[str, Tensor],
        output: dict[str, Tensor],
        timesteps: dict[str, Tensor],
        mask: Tensor | None = None,
        ligand_mask: Tensor | None = None,
        bond_matrix_gt: Tensor | None = None,
        protein_valid_mask: Tensor | None = None,
        ligand_valid_mask: Tensor | None = None,
        decoder_gt: dict[str, Tensor] | None = None,
        # DiffusionLoss params (for continuous structure)
        structure_embeddings_gt: Tensor | None = None,
        ligand_structure_embeddings_gt: Tensor | None = None,
        x_t_structure: Tensor | None = None,
        x_t_ligand_structure: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute multi-modal loss.

        Parameters
        ----------
        split : str
            Split name (train/val).
        x_gt : dict[str, Tensor]
            Ground truth tokens.
        output : dict[str, Tensor]
            Model output logits.
        timesteps : dict[str, Tensor]
            Timesteps for flow matching.
        mask : Tensor, optional
            Protein residue mask.
        ligand_mask : Tensor, optional
            Ligand atom mask.
        bond_matrix_gt : Tensor, optional
            Ground truth bond matrix.
        protein_valid_mask : Tensor, optional
            Mask for valid protein samples in mixed batch.
        ligand_valid_mask : Tensor, optional
            Mask for valid ligand samples in mixed batch.
        decoder_gt : dict[str, Tensor], optional
            Ground truth batch data for decoder loss (should contain coords, ligand_coords, etc.).
        structure_embeddings_gt : Tensor, optional
            Continuous structure embeddings for DiffusionLoss [B, L, D].
        ligand_structure_embeddings_gt : Tensor, optional
            Continuous ligand structure embeddings for DiffusionLoss [B, M, D].
        x_t_structure : Tensor, optional
            Interpolated structure tokens for mask derivation [B, L].
        x_t_ligand_structure : Tensor, optional
            Interpolated ligand structure tokens for mask derivation [B, M].

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            Total loss and loss dictionary.
        """
        total_loss = torch.tensor(0.0, device=output["sequence_logits"].device)
        loss_dict = {}

        # === PROTEIN LOSSES ===
        # Sequence loss (always discrete CE)
        loss_seq = self.interpolant_seq.loss(
            output["sequence_logits"], x_gt["sequence_tokens"], timesteps["sequence_tokens"]
        )
        if protein_valid_mask is not None:
            loss_seq = (loss_seq.mean(dim=-1) * protein_valid_mask.float()).sum() / (
                protein_valid_mask.float().sum() + 1e-8
            )
        else:
            loss_seq = loss_seq.mean()

        # Structure loss - Option A: Hybrid (DiffusionLoss if enabled)
        # Initialize predicted embeddings for later use in decoding
        pred_structure_embeddings = None
        pred_ligand_structure_embeddings = None

        if self.use_diffusion_loss_structure and structure_embeddings_gt is not None:
            # Use DiffusionLoss for structure tokens
            # Get hidden states for conditioning (need to expose from encoder)
            structure_hidden = output.get("protein_hidden_states", output.get("last_hidden_state"))
            if structure_hidden is not None:
                # Extract protein portion if full hidden state
                if "protein_hidden_states" not in output and mask is not None:
                    B, L = mask.shape
                    structure_hidden = structure_hidden[:, :L, :]

                # Create diffusion mask from flow matching: mask=1 where position was corrupted
                if x_t_structure is not None:
                    diffusion_mask = (x_t_structure == self.mask_index_struc_tokens).float()
                else:
                    # Fallback: all positions if interpolated tokens not available
                    diffusion_mask = torch.ones_like(mask).float()

                # Apply validity masks
                if protein_valid_mask is not None:
                    diffusion_mask = diffusion_mask * protein_valid_mask[:, None].float()
                if mask is not None:
                    diffusion_mask = diffusion_mask * mask.float()

                # Compute diffusion loss and get predicted embeddings for decoding
                loss_struc, pred_structure_embeddings = self.diffusion_loss_protein_struc(
                    target=structure_embeddings_gt,
                    z=structure_hidden,
                    mask=diffusion_mask,
                    return_pred=True,
                )
                loss_struc = loss_struc * self.diffusion_loss_weight
            else:
                # Fallback if hidden states not available
                logger.warning("No hidden states for DiffusionLoss, falling back to discrete loss")
                loss_struc = self.interpolant_struc.loss(
                    output["structure_logits"], x_gt["structure_tokens"], timesteps["structure_tokens"]
                ).mean()
        else:
            # Original discrete flow matching loss
            loss_struc = self.interpolant_struc.loss(
                output["structure_logits"], x_gt["structure_tokens"], timesteps["structure_tokens"]
            )
            if protein_valid_mask is not None:
                loss_struc = (loss_struc.mean(dim=-1) * protein_valid_mask.float()).sum() / (
                    protein_valid_mask.float().sum() + 1e-8
                )
            else:
                loss_struc = loss_struc.mean()

        total_loss = total_loss + loss_seq + loss_struc
        loss_dict[f"{split}_loss_seq"] = loss_seq
        loss_dict[f"{split}_loss_struc"] = loss_struc

        # === LIGAND LOSSES (if present) ===
        has_ligand = "ligand_atom_tokens" in x_gt and output["ligand_atom_logits"].shape[1] > 0

        if has_ligand:
            # Ligand atom type loss
            loss_lig_atom = self.interpolant_ligand_atom.loss(
                output["ligand_atom_logits"],
                x_gt["ligand_atom_tokens"],
                timesteps["ligand_atom_tokens"],
            )
            if ligand_valid_mask is not None:
                loss_lig_atom = (loss_lig_atom.mean(dim=-1) * ligand_valid_mask.float()).sum() / (
                    ligand_valid_mask.float().sum() + 1e-8
                )
            else:
                loss_lig_atom = loss_lig_atom.mean()

            # Ligand structure loss - Option A: Hybrid (DiffusionLoss if enabled)
            if self.use_diffusion_loss_structure and ligand_structure_embeddings_gt is not None:
                # Use DiffusionLoss for ligand structure
                ligand_hidden = output.get("ligand_hidden_states")
                if ligand_hidden is None:
                    # Extract from full hidden state
                    full_hidden = output.get("last_hidden_state")
                    if full_hidden is not None and mask is not None:
                        B, L = mask.shape
                        ligand_hidden = full_hidden[:, L:, :]

                if ligand_hidden is not None:
                    # Create diffusion mask from flow matching
                    if x_t_ligand_structure is not None:
                        lig_diffusion_mask = (x_t_ligand_structure == self.mask_index_struc_tokens).float()
                    else:
                        lig_diffusion_mask = torch.ones_like(ligand_mask).float()

                    # Apply validity masks
                    if ligand_valid_mask is not None:
                        lig_diffusion_mask = lig_diffusion_mask * ligand_valid_mask[:, None].float()
                    if ligand_mask is not None:
                        lig_diffusion_mask = lig_diffusion_mask * ligand_mask.float()

                    # Compute diffusion loss for ligand structure and get predicted embeddings
                    loss_lig_struc, pred_ligand_structure_embeddings = self.diffusion_loss_ligand_struc(
                        target=ligand_structure_embeddings_gt,
                        z=ligand_hidden,
                        mask=lig_diffusion_mask,
                        return_pred=True,
                    )
                    loss_lig_struc = loss_lig_struc * self.diffusion_loss_weight
                else:
                    # Fallback if hidden states not available
                    loss_lig_struc = self.interpolant_ligand_struc.loss(
                        output["ligand_structure_logits"],
                        x_gt["ligand_structure_tokens"],
                        timesteps["ligand_structure_tokens"],
                    ).mean()
            else:
                # Original discrete flow matching loss
                loss_lig_struc = self.interpolant_ligand_struc.loss(
                    output["ligand_structure_logits"],
                    x_gt["ligand_structure_tokens"],
                    timesteps["ligand_structure_tokens"],
                )
                if ligand_valid_mask is not None:
                    loss_lig_struc = (loss_lig_struc.mean(dim=-1) * ligand_valid_mask.float()).sum() / (
                        ligand_valid_mask.float().sum() + 1e-8
                    )
                else:
                    loss_lig_struc = loss_lig_struc.mean()

            # Bond prediction loss
            if bond_matrix_gt is not None:
                loss_bond = self.bond_loss_fn(output["bond_logits"], bond_matrix_gt, ligand_mask)
            else:
                loss_bond = torch.tensor(0.0, device=total_loss.device)

            total_loss = total_loss + (
                self.ligand_atom_loss_weight * loss_lig_atom
                + self.ligand_struct_loss_weight * loss_lig_struc
                + self.bond_loss_weight * loss_bond
            )

            loss_dict[f"{split}_loss_lig_atom"] = loss_lig_atom
            loss_dict[f"{split}_loss_lig_struc"] = loss_lig_struc
            loss_dict[f"{split}_loss_bond"] = loss_bond

        # === STRUCTURE DECODER LOSSES (if enabled) ===
        if self.decode_tokens_during_training and decoder_gt is not None:
            if mask is not None:
                with torch.no_grad():
                    # In continuous mode, use predicted embeddings from diffusion loss
                    # This allows tracking reconstruction learning during training
                    decode_output = output
                    if self.use_continuous_structure:
                        decode_output = dict(output)  # Copy to avoid modifying original
                        # Use predicted embeddings if available, otherwise fall back to ground truth
                        if pred_structure_embeddings is not None:
                            decode_output["structure_embeddings"] = pred_structure_embeddings
                        elif structure_embeddings_gt is not None:
                            decode_output["structure_embeddings"] = structure_embeddings_gt
                        if pred_ligand_structure_embeddings is not None:
                            decode_output["ligand_structure_embeddings"] = pred_ligand_structure_embeddings
                        elif ligand_structure_embeddings_gt is not None:
                            decode_output["ligand_structure_embeddings"] = ligand_structure_embeddings_gt

                    # Decode both protein and ligand together using vit_decoder
                    decoded_x = self.decode_structure(
                        decode_output, mask, ligand_mask=ligand_mask if has_ligand else None
                    )

                # Get the vit_decoder output
                vit_output = decoded_x.get("vit_decoder")

                # Check if output is a dict (protein + ligand) or tensor (protein only)
                if isinstance(vit_output, dict):
                    # Unified protein + ligand decode output
                    protein_coords = vit_output.get("protein_coords")
                    ligand_coords = vit_output.get("ligand_coords")

                    # Apply protein decoder loss (only if we have protein ground truth)
                    if protein_coords is not None and decoder_gt.get("coords_res") is not None:
                        protein_decoded_x = {"vit_decoder": protein_coords}
                        total_loss, loss_dict = self.apply_structure_decoder_loss(
                            split, decoder_gt, protein_decoded_x, mask, total_loss, loss_dict, prefix="protein_"
                        )

                    # Apply ligand decoder loss
                    if has_ligand and ligand_coords is not None and ligand_mask is not None:
                        ligand_decoder_gt = {
                            "ligand_coords": decoder_gt.get("ligand_coords"),
                            "coords_res": decoder_gt.get("coords_res"),  # For joint alignment in loss
                        }
                        ligand_decoded_x = {
                            "vit_decoder": {"ligand_coords": ligand_coords, "protein_coords": protein_coords}
                        }
                        # Create mask dict for ligand loss computation
                        mask_dict = {"protein_mask": mask, "ligand_mask": ligand_mask}
                        total_loss, loss_dict = self.apply_structure_decoder_loss(
                            split,
                            ligand_decoder_gt,
                            ligand_decoded_x,
                            mask_dict,
                            total_loss,
                            loss_dict,
                            prefix="ligand_",
                        )
                else:
                    # Protein-only output (backward compatible, only if we have protein ground truth)
                    if decoder_gt.get("coords_res") is not None:
                        total_loss, loss_dict = self.apply_structure_decoder_loss(
                            split, decoder_gt, decoded_x, mask, total_loss, loss_dict, prefix="protein_"
                        )

        return total_loss, loss_dict

    def apply_structure_decoder_loss(
        self,
        split: str,
        decoder_gt: dict[str, Tensor],
        decoded_x: dict[str, Tensor],
        mask: Tensor,
        total_loss: Tensor,
        loss_dict: dict[str, Tensor],
        just_loss: bool = False,
        keep_batch_dim: bool = False,
        prefix: str = "",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the structure decoder loss to the model for protein and/or ligand.

        Parameters
        ----------
        split : str
            Split name (train/val).
        decoder_gt : dict[str, Tensor]
            Ground truth decoder targets (e.g., coordinates).
        decoded_x : dict[str, Tensor]
            Decoded outputs from decoder_factory.
        mask : Tensor
            Mask for valid tokens.
        total_loss : Tensor
            Current accumulated loss.
        loss_dict : dict[str, Tensor]
            Dictionary to store individual losses.
        just_loss : bool
            If True, return only the loss without accumulating.
        keep_batch_dim : bool
            If True, keep batch dimension in loss computation.
        prefix : str
            Prefix for loss names (e.g., "protein_" or "ligand_").

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            Updated total loss and loss dictionary.
        """
        decoder_name = "vit_decoder"
        loss2apply = self.decoder_factory.get_loss(decoder_name)

        # Filter losses based on data type to avoid misleading metrics
        # Protein losses: l2_loss, pairwise_l2_loss
        # Ligand losses: ligand_l2_loss, ligand_pairwise_l2_loss
        if prefix == "protein_":
            loss2apply = [l for l in loss2apply if not l.startswith("ligand_")]
        elif prefix == "ligand_":
            loss2apply = [l for l in loss2apply if l.startswith("ligand_")]

        for loss2apply_ in loss2apply:
            loss = self.loss_factory(
                loss2apply_, decoder_gt, decoded_x[decoder_name], mask, keep_batch_dim=keep_batch_dim
            )
            if just_loss:
                return loss, loss_dict
            # Apply loss weighting from weight_dict in loss_factory; setting to 0 for now
            # as we need a different way to set weights for different losses
            # total_loss += self.loss_factory.weight_dict[loss2apply_] * loss
            total_loss += 0 * loss
            loss_dict[f"{split}_{prefix}{loss2apply_}"] = loss

        return total_loss, loss_dict

    def step(self, batch: dict[str, Tensor], batch_idx: int, split: Literal["train", "val"] = "train") -> dict:
        """Single training/validation step.

        Handles three cases:
        1. Protein-only batches (from PDB)
        2. Protein-ligand batches (from PDBBind, SAIR)
        3. Ligand-only batches (from GEOM) - still valuable for learning ligand conformations
        """
        # Get device from whatever tensor is available
        if "sequence" in batch:
            device = batch["sequence"].device
        elif "ligand_coords" in batch:
            device = batch["ligand_coords"].device
        else:
            device = next(self.parameters()).device

        # Set device for interpolants
        self.interpolant_seq.device = device
        self.interpolant_struc.device = device
        self.interpolant_ligand_atom.device = device
        self.interpolant_ligand_struc.device = device

        # Get validity masks if present (mixed batches)
        protein_valid_mask = batch.get("protein_valid_mask")
        ligand_valid_mask = batch.get("ligand_valid_mask")

        # Check if batch has protein and ligand data
        has_protein = "sequence" in batch and batch["sequence"].numel() > 0
        has_ligand = "ligand_coords" in batch and batch["ligand_coords"].numel() > 0

        # === APPLY SE(3) AUGMENTATION (training only) ===
        if self.use_se3_augmentation and split == "train":
            with torch.no_grad():
                protein_coords = batch.get("coords_res") if has_protein else None
                protein_mask = batch.get("mask") if has_protein else None
                ligand_coords = batch.get("ligand_coords") if has_ligand else None
                ligand_mask = batch.get("ligand_mask") if has_ligand else None

                if protein_coords is not None or ligand_coords is not None:
                    # Apply same SE(3) transform to protein and ligand jointly
                    augmented = apply_se3_augmentation_protein_ligand(
                        protein_coords=protein_coords.clone() if protein_coords is not None else None,
                        protein_mask=protein_mask,
                        ligand_coords=ligand_coords.clone() if ligand_coords is not None else None,
                        ligand_mask=ligand_mask,
                        random_se3=True,
                        translation_scale=self.se3_translation_scale,
                        backbone_noise=0.0,
                    )
                    if has_protein:
                        batch["coords_res"] = augmented.protein_coords
                    if has_ligand:
                        batch["ligand_coords"] = augmented.ligand_coords

        # === ENCODE GROUND TRUTH ===
        x_gt = {}
        mask = None
        seq_gt = None
        ligand_mask = None
        bond_matrix_gt = None
        structure_embeddings_gt = None
        ligand_structure_embeddings_gt = None

        with torch.no_grad():
            # Get data from batch
            protein_coords = batch.get("coords_res") if has_protein else None
            protein_mask = batch.get("mask") if has_protein else None
            protein_indices = batch.get("indices") if has_protein else None
            ligand_coords = batch.get("ligand_coords") if has_ligand else None
            ligand_mask = batch.get("ligand_mask") if has_ligand else None
            ligand_indices = batch.get("ligand_indices") if has_ligand else None
            ligand_atom_types = batch.get("ligand_element_indices") if has_ligand else None
            bond_matrix_gt = batch.get("bond_matrix") if has_ligand else None

            if has_protein:
                seq_gt = batch["sequence"]

            # Joint encoding when both protein and ligand are present
            if has_protein and has_ligand:
                encoded = self.encode_protein_ligand_structure(
                    protein_coords=protein_coords,
                    protein_mask=protein_mask,
                    protein_indices=protein_indices,
                    ligand_coords=ligand_coords,
                    ligand_mask=ligand_mask,
                    ligand_indices=ligand_indices,
                    ligand_atom_types=ligand_atom_types,
                    bond_matrix=bond_matrix_gt,
                )

                # Extract results
                struc_gt = encoded["protein_tokens"]
                mask = encoded["protein_mask"]
                structure_embeddings_gt = encoded["protein_embeddings"]

                ligand_struc_gt = encoded["ligand_tokens"]
                ligand_mask = encoded["ligand_mask"]
                ligand_structure_embeddings_gt = encoded["ligand_embeddings"]

                # Get ligand atom types
                ligand_atom_gt = (
                    ligand_atom_types if ligand_atom_types is not None else torch.full_like(ligand_indices, 3)
                )

                x_gt["sequence_tokens"] = seq_gt
                x_gt["structure_tokens"] = struc_gt
                x_gt["ligand_atom_tokens"] = ligand_atom_gt
                x_gt["ligand_structure_tokens"] = ligand_struc_gt

            elif has_protein:
                # Protein-only: use existing encode_structure method
                if self.use_diffusion_loss_structure:
                    result = self.encode_structure(
                        protein_coords, protein_mask, protein_indices, return_continuous=True
                    )
                    x_quant, _, mask, structure_embeddings_gt = result
                else:
                    x_quant, _, mask = self.encode_structure(protein_coords, protein_mask, protein_indices)

                struc_gt = torch.argmax(x_quant, dim=-1)
                struc_gt[~mask.bool()] = self.padding_index_struc_tokens

                x_gt["sequence_tokens"] = seq_gt
                x_gt["structure_tokens"] = struc_gt

            elif has_ligand:
                # Ligand-only: use existing encode_ligand_structure method
                if ligand_atom_types is None:
                    ligand_atom_gt = torch.full_like(ligand_indices, 3)
                else:
                    ligand_atom_gt = ligand_atom_types

                if self.use_diffusion_loss_structure:
                    ligand_struc_gt, _, ligand_structure_embeddings_gt = self.encode_ligand_structure(
                        ligand_coords, ligand_mask, ligand_indices, return_continuous=True
                    )
                else:
                    ligand_struc_gt, _ = self.encode_ligand_structure(ligand_coords, ligand_mask, ligand_indices)

                x_gt["ligand_atom_tokens"] = ligand_atom_gt
                x_gt["ligand_structure_tokens"] = ligand_struc_gt

        # Get batch size from available data
        if has_protein:
            B = seq_gt.shape[0]
            L = seq_gt.shape[1]
        else:
            # Ligand-only batch
            B = batch["ligand_coords"].shape[0]
            L = 1  # Dummy length for protein - we'll create empty protein tensors

        # For ligand-only batches, create dummy protein tensors
        if not has_protein:
            # Create minimal dummy protein data (will be masked out in loss)
            seq_gt = torch.full((B, L), self.pad_token_id, device=device, dtype=torch.long)
            struc_gt = torch.full((B, L), self.padding_index_struc_tokens, device=device, dtype=torch.long)
            mask = torch.zeros((B, L), device=device)
            x_gt["sequence_tokens"] = seq_gt
            x_gt["structure_tokens"] = struc_gt
            # Set protein_valid_mask to all zeros for ligand-only batches
            protein_valid_mask = torch.zeros(B, device=device, dtype=torch.bool)

        # === SAMPLE TIMESTEPS AND INTERPOLATE ===
        timesteps = self.get_timesteps(B, has_ligand=has_ligand)
        x_t = self.interpolate_tokens(x_gt, timesteps)

        # === CONDITIONING ===
        conditioning_tensor = torch.zeros((B, L, 1), device=device)

        # Get protein indices (or create dummy for ligand-only batches)
        if has_protein:
            residue_index = batch["indices"]
        else:
            residue_index = torch.zeros((B, L), device=device, dtype=torch.long)

        # === FORWARD ===
        output = self.forward(
            x_t,
            mask,
            residue_index,
            conditioning_tensor,
            timesteps=timesteps,
            ligand_mask=ligand_mask,
            bond_matrix=bond_matrix_gt,
            protein_valid_mask=protein_valid_mask,
            ligand_valid_mask=ligand_valid_mask,
        )

        # === COMPUTE LOSS ===
        # Only pass decoder_gt every 1000 steps to avoid overhead
        decoder_gt = batch if (self.decode_tokens_during_training and batch_idx % 1000 == 0) else None
        total_loss, loss_dict = self.compute_loss(
            split,
            x_gt,
            output,
            timesteps,
            mask=mask,
            ligand_mask=ligand_mask,
            bond_matrix_gt=bond_matrix_gt,
            protein_valid_mask=protein_valid_mask,
            ligand_valid_mask=ligand_valid_mask,
            # DiffusionLoss params
            structure_embeddings_gt=structure_embeddings_gt,
            ligand_structure_embeddings_gt=ligand_structure_embeddings_gt,
            x_t_structure=x_t.get("structure_tokens"),
            x_t_ligand_structure=x_t.get("ligand_structure_tokens"),
            decoder_gt=decoder_gt,
        )

        # Log metrics
        self.log_dict({f"{split}_loss": total_loss, **loss_dict}, batch_size=B)

        # === PREPARE OUTPUT (compatible with callbacks) ===
        # StructureDecodeCallback, InverseFoldingCallback, ProteinLigandDecodeCallback expect these keys
        outputs = {
            "loss": total_loss,
            "x_gt": x_gt,
            "output": output,
            # Timesteps (use dummy zeros for ligand-only batches)
            "train_timesteps_seq": timesteps.get("sequence_tokens", torch.zeros(B, device=device)),
            "train_timesteps_struc": timesteps.get("structure_tokens", torch.zeros(B, device=device)),
            # Conditioning
            "conditioning": 0,  # No conditioning used
            # Unmasked logits
            "unmasked_x": {
                "sequence_logits": output["sequence_logits"],
                "structure_logits": output["structure_logits"],
            },
            # Protein/Ligand info for callbacks
            "has_protein": has_protein,
            "has_ligand": has_ligand,
            "ligand_mask": ligand_mask,
            "mask": mask,
        }

        # Add ligand logits if present
        if has_ligand:
            outputs["ligand_atom_logits"] = output["ligand_atom_logits"]
            outputs["ligand_structure_logits"] = output["ligand_structure_logits"]
            outputs["bond_logits"] = output["bond_logits"]
            outputs["train_timesteps_ligand"] = timesteps.get("ligand_atom_tokens")

        # Decode structure (only during training, not every batch)
        if self.decode_tokens_during_training and batch_idx % 1000 == 0:
            with torch.no_grad():
                # In continuous mode, inject ground truth embeddings for decoding
                # (predicted embeddings are computed inside compute_loss but not exposed here)
                decode_output = output
                if self.use_continuous_structure:
                    decode_output = dict(output)  # Copy to avoid modifying original
                    if structure_embeddings_gt is not None:
                        decode_output["structure_embeddings"] = structure_embeddings_gt
                    if ligand_structure_embeddings_gt is not None:
                        decode_output["ligand_structure_embeddings"] = ligand_structure_embeddings_gt

                # Unified decode for both protein and ligand
                decoded_x = self.decode_structure(decode_output, mask, ligand_mask=ligand_mask if has_ligand else None)
                outputs["decoded_x"] = decoded_x

                # Extract ligand info (atom types, bond matrix) for callbacks
                if has_ligand:
                    decoded_ligand = self.extract_ligand_predictions(output, ligand_mask)
                    # Add ligand coords from unified decode output
                    vit_output = decoded_x.get("vit_decoder")
                    if isinstance(vit_output, dict) and "ligand_coords" in vit_output:
                        decoded_ligand["coords"] = vit_output["ligand_coords"]
                    outputs["decoded_ligand_x"] = decoded_ligand

        return outputs

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict:
        """Training step.

        Returns full output dict for callbacks (StructureDecodeCallback, InverseFoldingCallback).
        """
        result = self.step(batch, batch_idx, "train")
        # Must return dict with "loss" key for Lightning, plus callback-compatible keys
        return result

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Validation step."""
        result = self.step(batch, batch_idx, "val")
        return result["loss"]

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        # Use .get() instead of .pop() to avoid mutating scheduler_kwargs
        # This ensures configure_optimizers works correctly on checkpoint resumption
        scheduler_kwargs_copy = {
            k: v for k, v in self.scheduler_kwargs.items() if k not in ("num_training_steps", "num_warmup_steps")
        }
        scheduler = transformers.get_scheduler(
            self.scheduler,
            optimizer,
            num_training_steps=self.scheduler_kwargs.get("num_training_steps", None),
            num_warmup_steps=self.scheduler_kwargs.get("num_warmup_steps", None),
            scheduler_specific_kwargs=scheduler_kwargs_copy,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def generate_sample(
        self,
        length: int,
        num_samples: int,
        inference_schedule_seq: Callable = LogInferenceSchedule,
        inference_schedule_struc: Callable = LinearInferenceSchedule,
        inference_schedule_ligand_atom: Callable = None,
        inference_schedule_ligand_struc: Callable = None,
        nsteps: int = 200,
        stochasticity_seq: int = 20,
        stochasticity_struc: int = 20,
        temperature_seq: float = 0.5,
        temperature_struc: float = 1.0,
        inverse_folding: bool = False,
        forward_folding: bool = False,
        input_structure_coords: Tensor = None,
        input_sequence_tokens: Tensor = None,
        input_mask: Tensor = None,
        input_indices: Tensor = None,
        # Ligand generation params
        generate_ligand: bool = False,
        num_atoms: int = 30,
        input_ligand_atom_tokens: Tensor = None,
        input_ligand_structure_tokens: Tensor = None,
        input_ligand_structure_embeddings: Tensor = None,
        input_bond_matrix: Tensor = None,
        stochasticity_ligand: int = 20,
        temperature_ligand: float = 0.5,
        ligand_is_context: bool = False,
        # Compatibility with the base protein-only LeFlur (not used here)
        asynchronous_sampling: bool = False,
        # Diffusion loss sampling params (for continuous structure generation)
        diffusion_sampling_steps: int = 100,
        diffusion_temperature: float = 1.0,
    ):
        """Generate samples with optional ligand.

        This extends the protein-only LeFlur generation to support ligand
        generation alongside protein.

        When use_diffusion_loss_structure=True, structure generation uses
        DiffusionLoss.sample() to produce continuous embeddings, which are
        then decoded by vit_decoder to 3D coordinates.

        Parameters
        ----------
        inference_schedule_ligand_atom : Callable, optional
            Schedule class for ligand atom token generation. If None, falls back
            to inference_schedule_seq (protein sequence schedule).
        inference_schedule_ligand_struc : Callable, optional
            Schedule class for ligand structure token generation. If None, falls
            back to inference_schedule_struc (protein structure schedule).
        ligand_is_context : bool
            If True, the provided ligand is used as fixed conditioning context
            and will NOT be updated during generation. Use this for inverse/forward
            folding where the ligand provides structural context.
        input_ligand_structure_embeddings : Tensor, optional
            Continuous structure embeddings for ligand [B, N_atoms, D]. Required
            in continuous mode when ligand_is_context=True. Obtain these via
            encode_ligand_structure(..., return_continuous=True).
        """
        device = next(self.parameters()).device

        # Update interpolant devices (required for proper device handling during inference)
        self.interpolant_seq.device = device
        self.interpolant_struc.device = device
        if hasattr(self, "interpolant_ligand_atom"):
            self.interpolant_ligand_atom.device = device
        if hasattr(self, "interpolant_ligand_struc"):
            self.interpolant_ligand_struc.device = device

        # Initialize protein tokens
        xt_seq = self.interpolant_seq.sample_prior((num_samples, length)).to(device)
        xt_struc = self.interpolant_struc.sample_prior((num_samples, length)).to(device)

        # Set up schedules
        schedule_seq = inference_schedule_seq(nsteps=nsteps) if inference_schedule_seq else self.inference_schedule
        schedule_struc = (
            inference_schedule_struc(nsteps=nsteps) if inference_schedule_struc else self.inference_schedule
        )

        ts_seq = schedule_seq.generate_schedule(device=device)
        ts_struc = schedule_struc.generate_schedule(device=device)
        dts_seq = schedule_seq.discretize(device=device)
        dts_struc = schedule_struc.discretize(device=device)

        # Set up independent ligand schedules (fall back to protein schedules if not specified)
        schedule_lig_atom = (
            inference_schedule_ligand_atom(nsteps=nsteps) if inference_schedule_ligand_atom else schedule_seq
        )
        schedule_lig_struc = (
            inference_schedule_ligand_struc(nsteps=nsteps) if inference_schedule_ligand_struc else schedule_struc
        )
        ts_lig_atom = schedule_lig_atom.generate_schedule(device=device)
        ts_lig_struc = schedule_lig_struc.generate_schedule(device=device)
        dts_lig_atom = schedule_lig_atom.discretize(device=device)
        dts_lig_struc = schedule_lig_struc.discretize(device=device)

        # Initialize defaults (same as the protein-only LeFlur encoder)
        mask = torch.ones((num_samples, length), device=device)
        residue_index = torch.arange(length, device=device)
        conditioning_tensor = torch.zeros((num_samples, length, 1), device=device)

        # Handle inverse/forward folding
        # Use t=0.9950 (close to 1) to indicate "clean/conditioned" tokens
        # In discrete flow matching: t=0 is noise/mask, t=1 is clean data
        # Both can be True simultaneously to fix both protein sequence and structure
        # (e.g. for ligand-only refinement conditioned on a fixed protein)
        if inverse_folding and input_structure_coords is not None:
            x_quant, _, mask = self.encode_structure(input_structure_coords, input_mask, input_indices)
            xt_struc = x_quant.argmax(dim=-1).to(device)
            ts_struc = torch.full_like(ts_struc, 0.9950)  # Structure is clean/given (t≈1)
        if forward_folding and input_sequence_tokens is not None:
            xt_seq = input_sequence_tokens.to(device)
            ts_seq = torch.full_like(ts_seq, 0.9950)  # Sequence is clean/given (t≈1)

        # Initialize ligand tokens if generating
        xt_lig_atom = None
        xt_lig_struc = None
        ligand_mask = None
        bond_matrix = None

        if generate_ligand:
            ligand_mask = torch.ones((num_samples, num_atoms), device=device)
            if input_ligand_atom_tokens is not None:
                xt_lig_atom = input_ligand_atom_tokens.to(device)
            else:
                xt_lig_atom = self.interpolant_ligand_atom.sample_prior((num_samples, num_atoms)).to(device)

            if input_ligand_structure_tokens is not None:
                xt_lig_struc = input_ligand_structure_tokens.to(device)
            else:
                xt_lig_struc = self.interpolant_ligand_struc.sample_prior((num_samples, num_atoms)).to(device)

            if input_bond_matrix is not None:
                bond_matrix = input_bond_matrix.to(device)
            else:
                bond_matrix = None

        # Check if using continuous structure generation
        use_continuous_gen = self.use_diffusion_loss_structure and self.diffusion_loss_protein_struc is not None

        # For continuous mode: track which positions have been "committed" (generated)
        # Start with all positions needing generation
        if use_continuous_gen:
            # Initialize continuous embeddings storage
            protein_structure_embeddings = torch.zeros(
                num_samples, length, self.diffusion_loss_protein_struc.target_channels, device=device
            )
            protein_committed_mask = torch.zeros(
                num_samples, length, device=device
            )  # 0 = need to generate, 1 = committed

            if generate_ligand:
                # Check if ligand is fixed context with provided continuous embeddings
                if ligand_is_context and input_ligand_structure_embeddings is not None:
                    # Use provided embeddings and mark all positions as committed
                    ligand_structure_embeddings = input_ligand_structure_embeddings.to(device)
                    ligand_committed_mask = torch.ones(num_samples, num_atoms, device=device)
                else:
                    # Normal generation: start from zeros, commit as positions are unmasked
                    ligand_structure_embeddings = torch.zeros(
                        num_samples, num_atoms, self.diffusion_loss_ligand_struc.target_channels, device=device
                    )
                    ligand_committed_mask = torch.zeros(num_samples, num_atoms, device=device)

        # Generation loop (includes independent ligand schedules)
        for step_idx, (
            dt_seq,
            dt_struc,
            t_seq,
            t_struc,
            dt_lig_atom,
            dt_lig_struc,
            t_lig_atom,
            t_lig_struc,
        ) in enumerate(
            tqdm(
                zip(dts_seq, dts_struc, ts_seq, ts_struc, dts_lig_atom, dts_lig_struc, ts_lig_atom, ts_lig_struc),
                desc="Generating samples",
                total=len(dts_seq),
            )
        ):
            # Ensure all schedule tensors are on the correct device
            dt_seq = dt_seq.to(device)
            dt_struc = dt_struc.to(device)
            dt_lig_atom = dt_lig_atom.to(device)
            dt_lig_struc = dt_lig_struc.to(device)
            t_seq = schedule_seq.pad_time(num_samples, t_seq, device)
            t_struc = schedule_struc.pad_time(num_samples, t_struc, device)
            t_lig_atom = schedule_lig_atom.pad_time(num_samples, t_lig_atom, device)
            t_lig_struc = schedule_lig_struc.pad_time(num_samples, t_lig_struc, device)
            timesteps = {"sequence_tokens": t_seq, "structure_tokens": t_struc}

            x_t = {"sequence_tokens": xt_seq, "structure_tokens": xt_struc}
            if generate_ligand:
                timesteps["ligand_atom_tokens"] = t_lig_atom  # Independent ligand atom schedule
                timesteps["ligand_structure_tokens"] = t_lig_struc  # Independent ligand structure schedule
                x_t["ligand_atom_tokens"] = xt_lig_atom
                x_t["ligand_structure_tokens"] = xt_lig_struc

            # For MAR-style generation: pass previously sampled embeddings
            # The encoder will use these for committed positions, mask embedding for others
            structure_embeddings_for_forward = None
            ligand_structure_embeddings_for_forward = None

            if use_continuous_gen:
                # Create blended embeddings: sampled for committed, zeros for uncommitted
                # The mask token embedding will be added in encoder for uncommitted positions
                structure_embeddings_for_forward = protein_structure_embeddings * protein_committed_mask.unsqueeze(-1)
                if generate_ligand:
                    ligand_structure_embeddings_for_forward = (
                        ligand_structure_embeddings * ligand_committed_mask.unsqueeze(-1)
                    )

            output = self.forward(
                x_t,
                mask,
                residue_index,
                conditioning_tensor,
                timesteps=timesteps,
                ligand_mask=ligand_mask,
                bond_matrix=bond_matrix,
                structure_embeddings=structure_embeddings_for_forward,
                ligand_structure_embeddings=ligand_structure_embeddings_for_forward,
            )

            # Update protein sequence tokens (always discrete)
            xt_seq = self.interpolant_seq.step(
                output["sequence_logits"],
                t_seq,
                xt_seq,
                dt_seq,
                stochasticity=stochasticity_seq,
                temperature=temperature_seq,
            )

            # Update protein structure
            # Always run discrete interpolant step to get the masking schedule
            # This ensures continuous mode uses the SAME masking pattern as discrete mode
            prev_xt_struc = xt_struc.clone()
            xt_struc = self.interpolant_struc.step(
                output["structure_logits"],
                t_struc,
                xt_struc,
                dt_struc,
                stochasticity=stochasticity_struc,
                temperature=temperature_struc,
            )

            if use_continuous_gen:
                # Continuous mode: derive committed mask from discrete masking schedule
                # Positions that are no longer mask tokens are "committed"
                protein_hidden = output.get("protein_hidden_states")
                if protein_hidden is not None:
                    # Identify positions newly unmasked this step
                    # prev_xt_struc == mask_token AND xt_struc != mask_token
                    was_masked = prev_xt_struc == self.mask_index_struc_tokens
                    is_unmasked = xt_struc != self.mask_index_struc_tokens
                    newly_committed = was_masked & is_unmasked  # [B, L]

                    if newly_committed.any():
                        # Sample continuous embeddings for newly committed positions
                        with torch.no_grad():
                            sampled_embeddings = self.diffusion_loss_protein_struc.sample(
                                z=protein_hidden,
                                temperature=diffusion_temperature,
                                num_steps=diffusion_sampling_steps,
                            )

                        # Update embeddings and mask for newly committed positions
                        protein_structure_embeddings = torch.where(
                            newly_committed.unsqueeze(-1),
                            sampled_embeddings,
                            protein_structure_embeddings,
                        )
                        protein_committed_mask = torch.where(
                            newly_committed,
                            torch.ones_like(protein_committed_mask),
                            protein_committed_mask,
                        )

            # Update ligand tokens if generating (skip if ligand is fixed context)
            if generate_ligand and not ligand_is_context:
                xt_lig_atom = self.interpolant_ligand_atom.step(
                    output["ligand_atom_logits"],
                    t_lig_atom,  # Use independent ligand atom schedule
                    xt_lig_atom,
                    dt_lig_atom,  # Use independent ligand atom schedule
                    stochasticity=stochasticity_ligand,
                    temperature=temperature_ligand,
                )

                # Always run discrete interpolant step for ligand structure to get masking schedule
                prev_xt_lig_struc = xt_lig_struc.clone()
                xt_lig_struc = self.interpolant_ligand_struc.step(
                    output["ligand_structure_logits"],
                    t_lig_struc,  # Use independent ligand structure schedule
                    xt_lig_struc,
                    dt_lig_struc,  # Use independent ligand structure schedule
                    stochasticity=stochasticity_ligand,
                    temperature=temperature_ligand,
                )

                if use_continuous_gen:
                    # Continuous mode: derive committed mask from discrete masking schedule
                    ligand_hidden = output.get("ligand_hidden_states")
                    if ligand_hidden is not None:
                        # Identify positions newly unmasked this step
                        was_masked = prev_xt_lig_struc == self.mask_index_struc_tokens
                        is_unmasked = xt_lig_struc != self.mask_index_struc_tokens
                        newly_committed = was_masked & is_unmasked  # [B, N_atoms]

                        if newly_committed.any():
                            # Sample continuous embeddings for newly committed positions
                            with torch.no_grad():
                                sampled_lig_embeddings = self.diffusion_loss_ligand_struc.sample(
                                    z=ligand_hidden,
                                    temperature=diffusion_temperature,
                                    num_steps=diffusion_sampling_steps,
                                )

                            # Update embeddings and mask for newly committed positions
                            ligand_structure_embeddings = torch.where(
                                newly_committed.unsqueeze(-1),
                                sampled_lig_embeddings,
                                ligand_structure_embeddings,
                            )
                            ligand_committed_mask = torch.where(
                                newly_committed,
                                torch.ones_like(ligand_committed_mask),
                                ligand_committed_mask,
                            )

        # Final output
        result = output
        result["generated_seq_tokens"] = xt_seq

        if use_continuous_gen:
            # For continuous mode: return embeddings (decoding done by decode_structure())
            result["generated_structure_embeddings"] = protein_structure_embeddings
            result["generated_struc_tokens"] = None  # No discrete tokens
            # Also add as "structure_embeddings" for decode_structure() compatibility
            result["structure_embeddings"] = protein_structure_embeddings

            if generate_ligand:
                result["generated_ligand_structure_embeddings"] = ligand_structure_embeddings
                result["generated_ligand_struc_tokens"] = None
                # Also add as "ligand_structure_embeddings" for decode_structure() compatibility
                result["ligand_structure_embeddings"] = ligand_structure_embeddings
        else:
            result["generated_struc_tokens"] = xt_struc

        if generate_ligand:
            result["generated_ligand_atom_tokens"] = xt_lig_atom
            if not use_continuous_gen:
                result["generated_ligand_struc_tokens"] = xt_lig_struc
            result["predicted_bond_matrix"] = output["bond_logits"].argmax(dim=-1)

        return result
