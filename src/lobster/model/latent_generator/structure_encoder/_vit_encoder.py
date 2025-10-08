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

        self.n_atoms = n_atoms
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
        if "sequence" in batch:
            seq_mask = batch["mask"].clone()
            residue_index = batch["indices"].clone()
            coords = batch["coords_res"].clone()
            if self.use_sequence_embedding:
                sequence = batch["sequence"].clone()
            else:
                sequence = None
        else:
            seq_mask = None
            residue_index = None
            coords = None
            sequence = None

        if "ligand_coords" in batch:
            # need to figure out how to rotate and translate the ligand coords the same way as the protein coords
            ligand_coords = batch["ligand_coords"].clone()
            ligand_mask = batch["ligand_mask"].clone()
            ligand_residue_index = batch["ligand_indices"].clone()
            ligand_atomic_numbers = batch["ligand_atomic_numbers"].clone() if "ligand_atomic_numbers" in batch else None
            # combine protein and ligand coords but note index to splice out the ligand coords after rotation and translation
            if coords is None:
                coords = ligand_coords
                seq_mask = ligand_mask
                residue_index = ligand_residue_index
            else:
                B, L, n_atoms, _ = coords.shape
                coords = coords.reshape(B, -1, 3)
                coords = torch.cat([coords, ligand_coords], dim=1)
                seq_mask = torch.cat([seq_mask, ligand_mask], dim=1)

        frame_type = self.frame_type if frame_type is None else frame_type
        get_all_frames = self.get_all_frames if get_all_frames is None else get_all_frames
        apply_stochastic_fa = self.apply_stochastic_fa if apply_stochastic_fa is None else apply_stochastic_fa

        if random_se3:
            if only_rot:
                logger.info("only rotating")
                translation_scale = 0.0
            else:
                translation_scale = self.translation_scale
                if only_trans:
                    logger.info("only translating")
                    rotation_mode = "none"
                    coords = apply_random_se3_batched(
                        coords, translation_scale=translation_scale, rotation_mode=rotation_mode
                    )
                else:
                    coords = apply_random_se3_batched(coords, translation_scale=translation_scale)
        else:
            logger.info("no se3 applied")

        if frame_type is not None:
            # apply global frame
            coords = apply_global_frame_to_coords(
                coords,
                frame_type=frame_type,
                mask=seq_mask,
                apply_stochastic_fa=apply_stochastic_fa,
                get_all_frames=get_all_frames,
            )

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

        if "ligand_coords" in batch:
            if "sequence" in batch:
                # splice out the ligand coords
                ligand_coords = coords[:, L * n_atoms :, :]
                coords = coords[:, : L * n_atoms, :]
                coords = coords.reshape(B, L, n_atoms, 3)
                seq_mask = seq_mask[:, :L]
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
        return_embeddings: bool = False,
        **kwargs,
    ):
        if coords is not None:
            B, L, _, _ = coords.shape
            seq_mask[torch.isnan(seq_mask)] = 0
            coords = coords[:, :, : self.n_atoms, :]
        else:
            B, _, _ = ligand_coords.shape
        emb = self.net(
            coords,
            seq_mask=seq_mask,
            residue_index=residue_index,
            ligand_coords=ligand_coords,
            ligand_mask=ligand_mask,
            ligand_residue_index=ligand_residue_index,
            ligand_atom_types=ligand_atom_types,
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
