import torch
from torch import Tensor
from lobster.model.latent_generator.structure_encoder import BaseEncoder
from lobster.model.latent_generator.utils import apply_random_se3_batched
from lobster.model.latent_generator.models.vit._vit_utils import (
    PLMUViTEncoder,
    expand,
)


class PLMEncoder(BaseEncoder):
    """Wrapper for U-ViT module to encode structure coordinates."""

    def __init__(
        self,
        embed_dim: int,
        embed_dim_hidden: int,
        data_fixed_size: int,
        uvit_n_layers: int,
        uvit_n_heads: int,
        uvit_dim_head: int,
        uvit_position_embedding_type: str,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_sequential_to_out: bool = False,
        attn_drop_out_rate: float = 0.0,
        translation_scale: float = 1.0,
        use_template_coords: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.embed_dim_hidden = embed_dim_hidden
        self.attn_drop_out_rate = attn_drop_out_rate
        self.add_cls_token = False
        self.translation_scale = translation_scale
        self.use_template_coords = use_template_coords
        self.vit = PLMUViTEncoder(
            embed_dim=embed_dim,
            embed_dim_hidden=embed_dim_hidden,
            seq_len=data_fixed_size,
            depth=uvit_n_layers,
            heads=uvit_n_heads,
            dim_head=uvit_dim_head,
            position_embedding_type=uvit_position_embedding_type,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
            use_template_coords=use_template_coords,
        )

    def featurize(self, batch, translation_scale: float = None):
        seq_mask = batch["mask"].clone()
        residue_index = batch["indices"].clone()
        plm_embeddings = batch["plm_embeddings"].clone()
        if "template_coords" in batch and self.use_template_coords:
            coords = batch["template_coords"].clone()
            if translation_scale is None:
                translation_scale = self.translation_scale
            coords = apply_random_se3_batched(coords, translation_scale=translation_scale)
            coords[batch["template_mask"] == 0] = 0
        else:
            coords = None

        return plm_embeddings, seq_mask, residue_index, coords

    def forward(
        self,
        plm_embeddings: Tensor,
        seq_mask: Tensor,
        residue_index: Tensor | None = None,
        coords: Tensor | None = None,
        return_embeddings: bool = False,
        **kwargs,
    ):
        B, L, _ = plm_embeddings.shape
        seq_mask[torch.isnan(seq_mask)] = 0
        if not self.use_template_coords:
            coords = None

        emb = self.vit(
            plm_embeddings,
            seq_mask=seq_mask,
            residue_index=residue_index,
            coords=coords,
            attn_drop_out_rate=self.attn_drop_out_rate,
            return_embeddings=return_embeddings,
        )
        if return_embeddings:
            emb, emb_out = emb
        assert not torch.isnan(emb).any()

        emb *= expand(seq_mask, emb)

        if return_embeddings:
            return emb, emb_out
        else:
            return emb
