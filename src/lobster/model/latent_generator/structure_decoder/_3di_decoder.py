import torch
from torch import Tensor

from lobster.model.latent_generator.models.vit._vit_utils import expand
from ._decoder import BaseDecoder


class FoldSeek3diDecoder(BaseDecoder):
    """3di decoder that takes in the input tokens and outputs the 3di tokens"""

    def __init__(self, struc_token_codebook_size: int, out_token_codebook_size: int, *args, **kwargs):
        super().__init__()

        # Configuration
        self.struc_token_codebook_size = struc_token_codebook_size
        self.out_token_codebook_size = out_token_codebook_size

        # Neural network
        self.layer_norm = torch.nn.LayerNorm(self.struc_token_codebook_size)
        self.linear = torch.nn.Linear(self.struc_token_codebook_size, out_token_codebook_size, bias=False)

    def preprocess(self, coords: Tensor, mask: Tensor, **kwargs):
        return coords, mask

    def get_output_dim(self):
        return [self.out_token_codebook_size]

    def forward(
        self,
        x_quant: Tensor,
        seq_mask: Tensor,
        residue_index: Tensor | None = None,
        x_emb: Tensor = None,
        cls_token: bool = False,
        **kwargs,
    ):
        if isinstance(x_quant, dict):
            ligand_present = True
        else:
            ligand_present = False

        # check if we have cls token
        if cls_token:
            x_emb = x_emb[:, 1:, :]

        if ligand_present:
            x_quant = x_quant["protein_tokens"]
            seq_mask = seq_mask["protein_mask"]
            B, L = seq_mask.shape
            x_emb = x_emb[:, :L, :]

        # Create a copy of the mask to avoid in-place operations
        seq_mask = seq_mask.clone()
        seq_mask[torch.isnan(seq_mask)] = 0

        emb = self.layer_norm(x_emb)
        emb = self.linear(emb)
        assert not torch.isnan(emb).any()

        emb *= expand(seq_mask, emb)
        return emb
