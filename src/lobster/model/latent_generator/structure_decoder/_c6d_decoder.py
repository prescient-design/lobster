# ruff: noqa: F722
import torch
from lobster.model.latent_generator.structure_decoder import BaseDecoder

from torchtyping import TensorType


class C6DDecoder(BaseDecoder):
    """C6D decoder that takes in the input tokens and outputs the C6D predictions"""

    def __init__(self, struc_token_codebook_size: int, out_dim: int = 4, *args, **kwargs):
        super().__init__()

        # Configuration
        self.struc_token_codebook_size = struc_token_codebook_size
        self.out_dim = out_dim

        # Neural network
        self.layer_norm = torch.nn.LayerNorm(self.struc_token_codebook_size)
        # self.linear = torch.nn.Linear(self.struc_token_codebook_size, out_dim, bias=False)
        self.proj_symm = torch.nn.Linear(self.struc_token_codebook_size, 37 * 2)
        self.proj_asymm = torch.nn.Linear(self.struc_token_codebook_size, 37 + 19)

    def preprocess(self, coords: TensorType["b n a x", float], mask: TensorType["b n", float], **kwargs):
        return coords, mask

    def get_output_dim(self):
        return [self.out_dim]

    def forward(
        self,
        x_quant: TensorType["b n a x", float],
        seq_mask: TensorType["b n", float],
        residue_index: TensorType["b n", int] | None = None,
        x_emb: TensorType["b n a x", float] = None,
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
            x_quant = x_quant["protein_tokens"]
            seq_mask = seq_mask["protein_mask"]
            B, L = seq_mask.shape
            x_emb = x_emb[:, :L, :]

        # Create a copy of the mask to avoid in-place operations
        seq_mask = seq_mask.clone()
        seq_mask[torch.isnan(seq_mask)] = 0

        # Need to go from [b, n, d] to [b, n, n, d]
        x_emb = x_emb.unsqueeze(2)  # Shape: [b, n, 1, d]
        x_emb = x_emb.repeat(1, 1, x_emb.shape[1], 1)  # Repeat dim 2 to size n

        emb = self.layer_norm(x_emb)
        # emb = self.linear(emb)
        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(emb)
        logits_theta = logits_asymm[:, :, :, :37].permute(0, 3, 1, 2)
        logits_phi = logits_asymm[:, :, :, 37:].permute(0, 3, 1, 2)

        # predict dist, omega
        logits_symm = self.proj_symm(emb)
        logits_symm = logits_symm + logits_symm.permute(0, 2, 1, 3)
        logits_dist = logits_symm[:, :, :, :37].permute(0, 3, 1, 2)
        logits_omega = logits_symm[:, :, :, 37:].permute(0, 3, 1, 2)

        return [logits_dist, logits_omega, logits_theta, logits_phi]
