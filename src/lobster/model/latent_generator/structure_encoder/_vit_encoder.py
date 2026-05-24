import logging
from typing import Literal

import torch
from torch import Tensor

from lobster.model.latent_generator.models.vit._vit_utils import (
    TimeCondUViTEncoder,
    expand,
)
from lobster.model.latent_generator.utils import apply_global_frame_to_coords, apply_random_se3_batched

from ._encoder import BaseEncoder

logger = logging.getLogger(__name__)


class ViTEncoder(BaseEncoder):
    """Wrapper for U-ViT module to encode structure coordinates."""

    def __init__(
        self,
        embed_dim: int,
        embed_dim_hidden: int,
        data_fixed_size: int,
        n_atoms: int,
        uvit_n_layers: int,
        uvit_n_heads: int,
        uvit_dim_head: int,
        uvit_position_embedding_type: str,
        uvit_patch_size: int = 1,
        translation_scale: float = 1.0,
        backbone_noise=0.0,
        attn_bias_dim: int = None,
        pw_attn_bias: bool = False,
        concat_sine_pw: bool = True,
        attn_drop_out_rate=0.0,
        spatial_attention_mask=False,
        angstrom_cutoff=20,
        angstrom_cutoff_spatial=20.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        frame_type: Literal["norm_frame", "pca_frame", "mol_frame"] = None,
        apply_stochastic_fa: bool = False,
        get_all_frames: bool = False,
        use_sequential_to_out: bool = False,
        encode_ligand: bool = False,
        add_cls_token: bool = False,
        use_sequence_embedding: bool = False,
        mask_structure: float = 0.0,
        ligand_atom_embedding: bool = False,
        use_ligand_bond_embedding: bool = False,
        use_extended_element_vocab: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.backbone_noise = backbone_noise
        self.attn_bias_dim = attn_bias_dim
        self.pw_attn_bias = pw_attn_bias
        self.concat_sine_pw = concat_sine_pw
        logger.info(f"concat sine pw: {self.concat_sine_pw}")
        self.attn_drop_out_rate = attn_drop_out_rate
        self.angstrom_cutoff = angstrom_cutoff
        self.angstrom_cutoff_spatial = angstrom_cutoff_spatial
        logger.info(f"attention dropout rate: {self.attn_drop_out_rate}")
        self.spatial_attention_mask = spatial_attention_mask
        logger.info(
            f"spatial attention mask: {self.spatial_attention_mask} angstrom cutoff spatial: {self.angstrom_cutoff_spatial}"
        )
        self.mask_structure = mask_structure
        logger.info(f"mask structure {self.mask_structure} percentage")

        # Configuration
        self.translation_scale = translation_scale
        self.frame_type = frame_type
        self.apply_stochastic_fa = apply_stochastic_fa
        self.get_all_frames = get_all_frames
        logger.info(f"frame type: {self.frame_type}")
        logger.info(f"apply stochastic fa: {self.apply_stochastic_fa}")
        logger.info(f"get all frames: {self.get_all_frames}")
        self.encode_ligand = encode_ligand
        logger.info(f"encode ligand: {self.encode_ligand}")
        self.add_cls_token = add_cls_token
        logger.info(f"add cls token: {self.add_cls_token}")
        self.use_sequence_embedding = use_sequence_embedding
        logger.info(f"use sequence embedding: {self.use_sequence_embedding}")
        self.ligand_atom_embedding = ligand_atom_embedding
        logger.info(f"ligand atom embedding: {self.ligand_atom_embedding}")
        self.use_ligand_bond_embedding = use_ligand_bond_embedding
        logger.info(f"use ligand bond embedding: {self.use_ligand_bond_embedding}")
        self.use_extended_element_vocab = use_extended_element_vocab
        logger.info(f"use extended element vocab: {self.use_extended_element_vocab}")

        self.n_atoms = n_atoms
        self.embed_dim = embed_dim
        n_xyz = 3

        # Neural networks
        self.net = TimeCondUViTEncoder(
            embed_dim=embed_dim,
            embed_dim_hidden=embed_dim_hidden,
            seq_len=data_fixed_size,
            patch_size=uvit_patch_size,
            depth=uvit_n_layers,
            heads=uvit_n_heads,
            dim_head=uvit_dim_head,
            n_atoms=n_atoms,
            channels_per_atom=n_xyz,
            position_embedding_type=uvit_position_embedding_type,
            pw_attn_bias=self.pw_attn_bias,
            attn_bias_dim=self.attn_bias_dim,
            concat_sine_pw=self.concat_sine_pw,
            spatial_attention_mask=self.spatial_attention_mask,
            angstrom_cutoff=angstrom_cutoff,
            angstrom_cutoff_spatial=angstrom_cutoff_spatial,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
            encode_ligand=encode_ligand,
            add_cls_token=add_cls_token,
            sequence_embedding=use_sequence_embedding,
            ligand_atom_embedding=ligand_atom_embedding,
            use_ligand_bond_embedding=use_ligand_bond_embedding,
            use_extended_element_vocab=use_extended_element_vocab,
        )

    def featurize(
        self,
        batch,
        random_se3=True,
        only_rot=False,
        only_trans=False,
        frame_type: Literal["norm_frame", "pca_frame", "mol_frame"] = None,
        get_all_frames: bool = False,
        apply_stochastic_fa: bool = False,
        backbone_noise: float = None,
    ):
        # NEW: Extract validity masks (only present for heterogeneous batches)
        protein_valid = batch.get("protein_valid_mask", None)  # Shape: (batch_size,)
        ligand_valid = batch.get("ligand_valid_mask", None)  # Shape: (batch_size,)

        # Determine what data we have
        has_proteins = protein_valid is None or protein_valid.any()
        has_ligands = ligand_valid is None or ligand_valid.any()

        # Process protein data
        if has_proteins and "sequence" in batch:
            coords = batch["coords_res"].clone()
            seq_mask = batch["mask"].clone()
            residue_index = batch["indices"].clone()

            # NOTE: If protein_valid exists, some batch positions may be all-zero padding
            # The seq_mask will be False for those positions, so downstream processing
            # will naturally ignore them via masking

            if self.use_sequence_embedding:
                sequence = batch["sequence"].clone()
            else:
                sequence = None
        else:
            coords = None
            seq_mask = None
            residue_index = None
            sequence = None

        # Process ligand data
        if has_ligands and "ligand_coords" in batch:
            ligand_coords = batch["ligand_coords"].clone()
            ligand_mask = batch["ligand_mask"].clone()
            ligand_residue_index = batch["ligand_indices"].clone()
            ligand_atomic_numbers = batch["ligand_atomic_numbers"].clone() if "ligand_atomic_numbers" in batch else None

            # Combine protein and ligand if both present
            if coords is None:
                # Ligand-only case
                coords = ligand_coords
                seq_mask = ligand_mask
                residue_index = ligand_residue_index
            else:
                # Both present - concatenate (NOW SAFE with unified batch from Phase 1!)
                B, L, n_atoms, _ = coords.shape
                B_ligand = ligand_coords.shape[0]

                # Batch sizes MUST match with unified batch approach from Phase 1
                assert B == B_ligand, f"Batch size mismatch: protein {B} vs ligand {B_ligand}"

                # Flatten protein coords and concatenate
                coords = coords.reshape(B, -1, 3)  # [B, L*n_atoms, 3]
                coords = torch.cat([coords, ligand_coords], dim=1)  # [B, L*n_atoms + L_ligand, 3]

                # Expand seq_mask to match flattened protein coords: [B, L] -> [B, L*n_atoms]
                seq_mask = torch.cat(
                    [seq_mask.unsqueeze(-1).expand(-1, -1, n_atoms).reshape(B, -1), ligand_mask], dim=1
                )  # [B, L*n_atoms + L_ligand]
                # seq_mask = torch.cat([seq_mask, ligand_mask], dim=1)  # [B, L + L_ligand]

                # NOTE: For batch positions where ligand_valid=False:
                #   - ligand_coords[i] is all zeros (from collate padding)
                #   - ligand_mask[i] is all False (from collate padding)
                # For batch positions where protein_valid=False:
                #   - coords[i] is all zeros (from collate padding)
                #   - seq_mask[i] is all False (from collate padding)
                # The masks handle this naturally!
        else:
            ligand_coords = None
            ligand_mask = None
            ligand_residue_index = None
            ligand_atomic_numbers = None

        frame_type = self.frame_type if frame_type is None else frame_type
        get_all_frames = self.get_all_frames if get_all_frames is None else get_all_frames
        apply_stochastic_fa = self.apply_stochastic_fa if apply_stochastic_fa is None else apply_stochastic_fa

        # Apply SE(3) transformations - only if we have valid data
        # Pass atom_mask to ensure we only transform non-masked (valid) regions
        if random_se3 and coords is not None:
            # Check if we have any valid coordinates to transform
            if seq_mask is not None and seq_mask.any():
                if only_rot:
                    logger.info("only rotating")
                    translation_scale = 0.0
                else:
                    translation_scale = self.translation_scale
                    if only_trans:
                        logger.info("only translating")
                        rotation_mode = "none"
                        coords = apply_random_se3_batched(
                            coords, atom_mask=seq_mask, translation_scale=translation_scale, rotation_mode=rotation_mode
                        )
                    else:
                        coords = apply_random_se3_batched(
                            coords, atom_mask=seq_mask, translation_scale=translation_scale
                        )
            else:
                logger.debug("Skipping SE(3) transform - no valid coordinates")
        else:
            if not random_se3:
                logger.info("no se3 applied")

        if frame_type is not None and coords is not None:
            # Apply global frame only if we have valid coordinates
            if seq_mask is not None and seq_mask.any():
                coords = apply_global_frame_to_coords(
                    coords,
                    frame_type=frame_type,
                    mask=seq_mask,  # Mask handles padded positions
                    apply_stochastic_fa=apply_stochastic_fa,
                    get_all_frames=get_all_frames,
                )
            else:
                logger.debug("Skipping frame application - no valid coordinates")

        if self.backbone_noise > 0 and backbone_noise is None:
            coords = coords + self.backbone_noise * torch.randn_like(coords)
        elif backbone_noise is not None:
            coords = coords + backbone_noise * torch.randn_like(coords)
        else:
            coords = coords

        if self.mask_structure > 0:
            mask_structure = torch.rand(coords.shape[0], coords.shape[1], device=coords.device) > self.mask_structure
            if len(coords.shape) == 3:
                coords = coords * mask_structure.unsqueeze(-1)
            else:
                coords = coords * mask_structure.unsqueeze(-1).unsqueeze(-1)

        if has_ligands and "ligand_coords" in batch:
            if has_proteins and "sequence" in batch and coords is not None:
                # Both present - split them back out
                # NOTE: With unified batch, both modalities exist for all batch positions
                # The masks (seq_mask, ligand_mask) handle which are valid
                ligand_coords = coords[:, L * n_atoms :, :]
                coords = coords[:, : L * n_atoms, :]
                coords = coords.reshape(B, L, n_atoms, 3)
                # seq_mask = seq_mask[:, :L]  # Keep only protein mask
                # keep only protein mask and make it just a residue mask so from [B, L*n_atoms+L_ligand] to [B, L]
                seq_mask = seq_mask[:, : L * n_atoms].reshape(B, L, n_atoms)
                seq_mask = seq_mask.sum(dim=-1) > 0

                # ligand_mask = seq_mask[:, L*n_atoms:]

                return (
                    coords,
                    seq_mask,
                    residue_index,
                    sequence,
                    ligand_coords,
                    ligand_mask,
                    ligand_residue_index,
                    ligand_atomic_numbers,
                )
            else:
                # Ligand-only
                return None, None, None, None, ligand_coords, ligand_mask, ligand_residue_index, ligand_atomic_numbers

        return coords, seq_mask, residue_index, sequence

    def forward(
        self,
        coords: Tensor,
        seq_mask: Tensor,
        residue_index: Tensor | None = None,
        sequence: Tensor | None = None,
        ligand_coords: Tensor | None = None,
        ligand_mask: Tensor | None = None,
        ligand_residue_index: Tensor | None = None,
        ligand_atom_types: Tensor | None = None,
        ligand_bond_matrix: Tensor | None = None,
        return_embeddings: bool = False,
        **kwargs,
    ):
        if coords is not None:
            B, L, _, _ = coords.shape
            seq_mask[torch.isnan(seq_mask)] = 0
            coords = coords[:, :, : self.n_atoms, :]
        else:
            B, _, _ = ligand_coords.shape

        # Extract bond_matrix from batch if not passed explicitly
        if ligand_bond_matrix is None and "batch" in kwargs:
            batch = kwargs["batch"]
            if batch is not None and "bond_matrix" in batch:
                ligand_bond_matrix = batch["bond_matrix"]

        emb = self.net(
            coords,
            seq_mask=seq_mask,
            residue_index=residue_index,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            ligand_residue_index=ligand_residue_index,
            ligand_atom_types=ligand_atom_types,
            ligand_bond_matrix=ligand_bond_matrix,
            attn_drop_out_rate=self.attn_drop_out_rate,
            return_embeddings=return_embeddings,
            sequence=sequence,
        )
        if return_embeddings:
            emb, emb_out = emb

        assert not torch.isnan(emb).any()

        if ligand_coords is not None:
            if coords is not None:
                seq_mask = torch.cat([seq_mask, ligand_mask], -1)
            else:
                seq_mask = ligand_mask

        if self.add_cls_token:
            seq_mask = torch.cat([torch.ones(B, 1, device=emb.device), seq_mask], dim=1)

        emb *= expand(seq_mask, emb)

        if return_embeddings:
            return emb, emb_out
        else:
            return emb
