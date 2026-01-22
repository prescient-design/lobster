"""Gen-UME Protein-Ligand Lightning Module.

This module provides the PyTorch Lightning training wrapper for the
Gen-UME protein-ligand encoder. It handles:
- Multi-modal loss computation (protein_seq, protein_struct, ligand_atom, ligand_struct, bond)
- Mixed batch handling with validity masks
- Independent time sampling for different modalities
- Structure encoding/decoding via LatentGenerator

Key features:
- Backward compatible with protein-only training
- Supports mixed protein-only + protein-ligand batches
- Flow matching for all modalities
"""

import logging
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import transformers
from lightning import LightningModule
from torch import Tensor
from tqdm import tqdm

from lobster.model.latent_generator.cmdline import LatentEncoderDecoder
from lobster.model.latent_generator.cmdline import methods as latent_generator_methods
from lobster.model.latent_generator.utils.residue_constants import NUM_BOND_TYPES

from ._bond_prediction import BondMatrixLoss
from ._gen_ume_protein_ligand_encoder import AuxiliaryTask, ProteinLigandEncoderModule

# Bionemo interpolant code
from bionemo.moco.distributions.prior import DiscreteMaskedPrior, DiscreteUniformPrior
from bionemo.moco.distributions.time import UniformTimeDistribution
from bionemo.moco.interpolants import DiscreteFlowMatcher
from bionemo.moco.schedules.inference_time_schedules import LinearInferenceSchedule, LogInferenceSchedule

logger = logging.getLogger(__name__)


class ProteinLigandEncoderLightningModule(LightningModule):
    """PyTorch Lightning module for Gen-UME protein-ligand training.

    This module extends the original Gen-UME to support both protein-only
    and protein-ligand tasks. When no ligand data is present in a batch,
    it behaves identically to the original Gen-UME.

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

        # Auxiliary tasks
        self.auxiliary_tasks = auxiliary_tasks
        self.auxiliary_task_loss_fns = {"regression": nn.MSELoss()}

        # LatentGenerator for structure encoding/decoding
        self.decode_tokens_during_training = decode_tokens_during_training
        self.structure_latent_encoder_decoder = LatentEncoderDecoder()
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

        # === FLOW MATCHING SETUP ===
        self.inverse_folding = inverse_folding
        self.mask_index_struc_tokens = self.quantizer.n_tokens
        self.padding_index_struc_tokens = self.quantizer.n_tokens + 1
        self.num_struc_classes = self.quantizer.n_tokens + 2

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
        self.encoder = ProteinLigandEncoderModule(
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

    def encode_structure(self, x_gt: Tensor, mask: Tensor, residue_index: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode protein structure to tokens using LatentGenerator.

        Handles both FSQLigandTokenizer (returns dicts) and standard quantizers.

        Returns
        -------
        x_quant : Tensor
            Soft token distribution with shape [B, L, n_tokens] for FSQLigandTokenizer,
            or quantized logits for standard quantizers.
        x_quant_emb : Tensor
            Embeddings from encoder.
        out_mask : Tensor
            Mask for valid positions.
        """
        x_emb = self.structure_encoder(x_gt, mask, residue_index=residue_index)

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

        return x_quant, x_quant_emb, out_mask

    def encode_ligand_structure(
        self, ligand_coords: Tensor, ligand_mask: Tensor, atom_indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode ligand structure to tokens using LatentGenerator.

        Note: The ViT encoder accepts ligand_coords as a separate argument with shape [B, N_atoms, 3].

        Returns
        -------
        x_quant_argmax : Tensor
            Structure token indices with shape [B, N_atoms].
        out_mask : Tensor
            Mask for valid positions.
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

        return x_quant_argmax, out_mask

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

        Parameters
        ----------
        generated_output : dict
            Output from forward pass or generate_sample, containing structure_logits
            and optionally ligand_structure_logits
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

        # Get protein structure logits
        if "structure_logits" in generated_output:
            struc_logits = generated_output["structure_logits"]
        else:
            raise ValueError("No structure_logits found in output")

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
            lig_struc_tokens = lig_struc_logits[..., : self.quantizer.n_tokens]
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
    ) -> dict[str, Tensor]:
        """Forward pass through encoder."""
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

        Returns
        -------
        tuple[Tensor, dict[str, Tensor]]
            Total loss and loss dictionary.
        """
        total_loss = torch.tensor(0.0, device=output["sequence_logits"].device)
        loss_dict = {}

        # === PROTEIN LOSSES ===
        # Sequence loss
        loss_seq = self.interpolant_seq.loss(
            output["sequence_logits"], x_gt["sequence_tokens"], timesteps["sequence_tokens"]
        )
        if protein_valid_mask is not None:
            loss_seq = (loss_seq.mean(dim=-1) * protein_valid_mask.float()).sum() / (
                protein_valid_mask.float().sum() + 1e-8
            )
        else:
            loss_seq = loss_seq.mean()

        # Structure loss
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

            # Ligand structure loss
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
                    # Decode both protein and ligand together using vit_decoder
                    decoded_x = self.decode_structure(output, mask, ligand_mask=ligand_mask if has_ligand else None)

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

        # === ENCODE GROUND TRUTH ===
        x_gt = {}
        mask = None
        seq_gt = None
        ligand_mask = None
        bond_matrix_gt = None

        with torch.no_grad():
            # Protein (if present)
            if has_protein:
                seq_gt = batch["sequence"]
                # Use coords_res from collate function
                protein_coords = batch.get("coords_res", batch.get("input", [None])[0])
                x_quant, _, mask = self.encode_structure(protein_coords, batch["mask"], batch["indices"])
                struc_gt = torch.argmax(x_quant, dim=-1)
                struc_gt[~mask.bool()] = self.padding_index_struc_tokens

                x_gt["sequence_tokens"] = seq_gt
                x_gt["structure_tokens"] = struc_gt

            # Ligand (if present)
            if has_ligand:
                ligand_coords = batch["ligand_coords"]
                ligand_mask = batch["ligand_mask"]
                ligand_indices = batch["ligand_indices"]

                # Get ligand atom types
                if "ligand_element_indices" in batch:
                    ligand_atom_gt = batch["ligand_element_indices"]
                else:
                    # Fallback to all carbon if not provided
                    ligand_atom_gt = torch.full_like(ligand_indices, 3)  # 3 = C in ELEMENT_VOCAB_EXTENDED

                # Encode ligand structure
                ligand_struc_gt, _ = self.encode_ligand_structure(ligand_coords, ligand_mask, ligand_indices)

                x_gt["ligand_atom_tokens"] = ligand_atom_gt
                x_gt["ligand_structure_tokens"] = ligand_struc_gt

                # Get bond matrix
                bond_matrix_gt = batch.get("bond_matrix")

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
                # Unified decode for both protein and ligand
                decoded_x = self.decode_structure(output, mask, ligand_mask=ligand_mask if has_ligand else None)
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
        input_bond_matrix: Tensor = None,
        stochasticity_ligand: int = 20,
        temperature_ligand: float = 0.5,
    ):
        """Generate samples with optional ligand.

        This extends the original Gen-UME generation to support ligand
        generation alongside protein.
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

        # Initialize defaults (same as original Gen-UME)
        mask = torch.ones((num_samples, length), device=device)
        residue_index = torch.arange(length, device=device)
        conditioning_tensor = torch.zeros((num_samples, length, 1), device=device)

        # Handle inverse/forward folding
        if inverse_folding and input_structure_coords is not None:
            x_quant, _, mask = self.encode_structure(input_structure_coords, input_mask, input_indices)
            xt_struc = x_quant.argmax(dim=-1).to(device)
            ts_struc = torch.full_like(ts_struc, 0.9950)
        elif forward_folding and input_sequence_tokens is not None:
            xt_seq = input_sequence_tokens.to(device)
            ts_seq = torch.full_like(ts_seq, 0.9950)

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

        # Generation loop
        for dt_seq, dt_struc, t_seq, t_struc in tqdm(
            zip(dts_seq, dts_struc, ts_seq, ts_struc), desc="Generating samples"
        ):
            # Ensure all schedule tensors are on the correct device
            dt_seq = dt_seq.to(device)
            dt_struc = dt_struc.to(device)
            t_seq = schedule_seq.pad_time(num_samples, t_seq, device)
            t_struc = schedule_struc.pad_time(num_samples, t_struc, device)
            timesteps = {"sequence_tokens": t_seq, "structure_tokens": t_struc}

            x_t = {"sequence_tokens": xt_seq, "structure_tokens": xt_struc}
            if generate_ligand:
                timesteps["ligand_atom_tokens"] = t_seq  # Use same timestep for simplicity
                timesteps["ligand_structure_tokens"] = t_struc
                x_t["ligand_atom_tokens"] = xt_lig_atom
                x_t["ligand_structure_tokens"] = xt_lig_struc

            output = self.forward(
                x_t,
                mask,
                residue_index,
                conditioning_tensor,
                timesteps=timesteps,
                ligand_mask=ligand_mask,
                bond_matrix=bond_matrix,
            )

            # Update protein tokens
            xt_seq = self.interpolant_seq.step(
                output["sequence_logits"],
                t_seq,
                xt_seq,
                dt_seq,
                stochasticity=stochasticity_seq,
                temperature=temperature_seq,
            )
            xt_struc = self.interpolant_struc.step(
                output["structure_logits"],
                t_struc,
                xt_struc,
                dt_struc,
                stochasticity=stochasticity_struc,
                temperature=temperature_struc,
            )

            # Update ligand tokens if generating
            if generate_ligand:
                xt_lig_atom = self.interpolant_ligand_atom.step(
                    output["ligand_atom_logits"],
                    t_seq,
                    xt_lig_atom,
                    dt_seq,
                    stochasticity=stochasticity_ligand,
                    temperature=temperature_ligand,
                )
                xt_lig_struc = self.interpolant_ligand_struc.step(
                    output["ligand_structure_logits"],
                    t_struc,
                    xt_lig_struc,
                    dt_struc,
                    stochasticity=stochasticity_ligand,
                    temperature=temperature_ligand,
                )

        # Final output
        result = output
        result["generated_seq_tokens"] = xt_seq
        result["generated_struc_tokens"] = xt_struc
        if generate_ligand:
            result["generated_ligand_atom_tokens"] = xt_lig_atom
            result["generated_ligand_struc_tokens"] = xt_lig_struc
            result["predicted_bond_matrix"] = output["bond_logits"].argmax(dim=-1)

        return result
