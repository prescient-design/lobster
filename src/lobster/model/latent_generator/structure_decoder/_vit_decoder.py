import torch
from torch import Tensor

from lobster.model.latent_generator.models.vit._vit_utils import (
    TimeCondUViTDecoder,
    expand,
)

from ._decoder import BaseDecoder


class ViTDecoder(BaseDecoder):
    """Wrapper for U-ViT module to decode structure coordinates."""

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
        *args,
        **kwargs,
    ):
        super().__init__()

        # Configuration
        self.translation_scale = translation_scale

        self.n_atoms = n_atoms
        self.refinement_module = refinement_module

        # Neural networks
        self.net = TimeCondUViTDecoder(
            struc_token_codebook_size=struc_token_codebook_size,
            struc_token_dim=struc_token_dim,
            seq_len=data_fixed_size,
            patch_size=uvit_patch_size,
            depth=uvit_n_layers,
            heads=uvit_n_heads,
            dim_head=uvit_dim_head,
            n_atoms=n_atoms,
            position_embedding_type=uvit_position_embedding_type,
            indexed=indexed,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
            encode_ligand=encode_ligand,
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
        **kwargs,
    ):
        if isinstance(x_quant, dict):
            ligand_present = True
        else:
            ligand_present = False

        if ligand_present:
            ligand_quant = x_quant["ligand_tokens"]
            ligand_mask = seq_mask["ligand_mask"]
            if "protein_tokens" in x_quant:
                x_quant = x_quant["protein_tokens"]
                seq_mask = seq_mask["protein_mask"]
            else:
                x_quant = None
                seq_mask = None
        else:
            ligand_quant = None
            ligand_mask = None

        if seq_mask is not None:
            seq_mask[torch.isnan(seq_mask)] = 0

        emb = self.net(
            x_quant,
            seq_mask=seq_mask,
            residue_index=residue_index,
            ligand_quant=ligand_quant,
            ligand_mask=ligand_mask,
            **kwargs,
        )

        if ligand_present:
            ligand_emb = emb["ligand_coords"]
            emb = emb["protein_coords"]
            assert not torch.isnan(ligand_emb).any()
            ligand_emb *= expand(ligand_mask, ligand_emb)

        if self.refinement_module:
            emb_refinement = emb["protein_coords_refinement"]
            emb = emb["protein_coords"]
            assert not torch.isnan(emb_refinement).any()
            emb_refinement *= expand(seq_mask, emb_refinement)

        if x_quant is not None:
            assert not torch.isnan(emb).any()
            emb *= expand(seq_mask, emb)

        if ligand_present:
            out = {"protein_coords": emb, "ligand_coords": ligand_emb}
        elif self.refinement_module:
            out = {"protein_coords": emb, "protein_coords_refinement": emb_refinement}
        else:
            out = emb

        return out
