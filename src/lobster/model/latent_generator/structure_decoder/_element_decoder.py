import torch
from torch import Tensor
from lobster.model.latent_generator.structure_decoder import BaseDecoder
from lobster.model.latent_generator.models.vit._vit_utils import expand


class ElementDecoder(BaseDecoder):
    """Element decoder that takes in the input tokens and outputs element type tokens for ligands"""

    def __init__(self, struc_token_codebook_size: int, out_token_codebook_size: int = 14):
        super().__init__()

        # Configuration
        self.struc_token_codebook_size = struc_token_codebook_size
        self.out_token_codebook_size = out_token_codebook_size  # Element vocabulary size

        # Neural network
        self.layer_norm = torch.nn.LayerNorm(self.struc_token_codebook_size)
        self.linear = torch.nn.Linear(self.struc_token_codebook_size, out_token_codebook_size, bias=False)

    def preprocess(self, coords: Tensor, mask: Tensor):
        return coords, mask

    def get_output_dim(self):
        return [self.out_token_codebook_size]

    def forward(
        self,
        x_quant: Tensor,
        seq_mask: Tensor,
        x_emb: Tensor = None,
        cls_token: bool = False,
        **kwargs,
    ):
        if isinstance(x_quant, dict):
            ligand_present = True
        else:
            ligand_present = False

        if cls_token:
            x_emb = x_emb[:, 1:, :]

        if ligand_present:
            # Extract ligand tokens and mask for element prediction
            x_quant = x_quant["ligand_tokens"]
            # Get protein length before extracting ligand mask
            L_protein = seq_mask["protein_mask"].shape[1] if "protein_mask" in seq_mask else 0
            seq_mask = seq_mask["ligand_mask"]
            L_ligand = seq_mask.shape[1]
            # Extract ligand portion from embeddings
            x_emb = x_emb[:, L_protein : L_protein + L_ligand, :]
        else:
            # No ligands present, return zero tensor
            B = x_emb.shape[0] if x_emb is not None else 1
            return torch.zeros(B, 0, self.out_token_codebook_size, device=x_emb.device if x_emb is not None else "cpu")

        # Create a copy of the mask to avoid in-place operations
        seq_mask = seq_mask.clone()
        seq_mask[torch.isnan(seq_mask)] = 0

        emb = self.layer_norm(x_emb)
        emb = self.linear(emb)
        assert not torch.isnan(emb).any()

        emb *= expand(seq_mask, emb)
        return emb
