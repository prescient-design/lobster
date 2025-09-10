# ruff: noqa: F722
import torch
from lobster.model.latent_generator.structure_decoder import BaseDecoder
from torchtyping import TensorType
from lobster.model.latent_generator.models.vit._vit_utils import expand


class RgDecoder(BaseDecoder):
    """Radius of gyration decoder that predicts a single scalar value per protein structure."""

    def __init__(self, struc_token_codebook_size: int, *args, **kwargs):
        super().__init__()

        # Configuration
        self.struc_token_codebook_size = struc_token_codebook_size

        # Neural network - simple like sequence decoder
        self.layer_norm = torch.nn.LayerNorm(self.struc_token_codebook_size)
        self.linear = torch.nn.Linear(self.struc_token_codebook_size, 1, bias=False)

    def preprocess(self, coords: TensorType["b n a x", float], mask: TensorType["b n", float], **kwargs):
        return coords, mask

    def get_output_dim(self):
        return [1]

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

        # Handle different mask formats
        if ligand_present:
            # Extract appropriate mask based on data type
            if isinstance(seq_mask, dict):
                if "protein_mask" in seq_mask:
                    # Protein-ligand complex case
                    seq_mask = seq_mask["protein_mask"]
                elif "ligand_mask" in seq_mask:
                    # Ligand-only case
                    seq_mask = seq_mask["ligand_mask"]
                else:
                    raise ValueError(f"Unknown seq_mask keys: {list(seq_mask.keys())}")

            B, L = seq_mask.shape
            x_emb = x_emb[:, :L, :]

        # Create a copy of the mask to avoid in-place operations
        seq_mask = seq_mask.clone()
        seq_mask[torch.isnan(seq_mask)] = 0

        emb = self.layer_norm(x_emb)
        emb = self.linear(emb)  # (B, L, 1)
        assert not torch.isnan(emb).any()

        # Global sum pooling for Rg (single value per structure)
        emb_masked = emb * expand(seq_mask, emb)  # Apply mask
        pooled = emb_masked.sum(dim=1)  # (B, 1) - sum instead of average

        return pooled.squeeze(-1)  # (B,)


class SasaDecoder(BaseDecoder):
    """Solvent Accessible Surface Area decoder that predicts a single scalar value per protein structure."""

    def __init__(self, struc_token_codebook_size: int, *args, **kwargs):
        super().__init__()

        # Configuration
        self.struc_token_codebook_size = struc_token_codebook_size

        # Neural network - simple like RgDecoder
        self.layer_norm = torch.nn.LayerNorm(self.struc_token_codebook_size)
        self.linear = torch.nn.Linear(self.struc_token_codebook_size, 1, bias=False)

    def preprocess(self, coords: TensorType["b n a x", float], mask: TensorType["b n", float], **kwargs):
        return coords, mask

    def get_output_dim(self):
        return [1]

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

        # Handle different mask formats
        if ligand_present:
            # Extract appropriate mask based on data type
            if isinstance(seq_mask, dict):
                if "protein_mask" in seq_mask:
                    # Protein-ligand complex case
                    seq_mask = seq_mask["protein_mask"]
                elif "ligand_mask" in seq_mask:
                    # Ligand-only case
                    seq_mask = seq_mask["ligand_mask"]
                else:
                    raise ValueError(f"Unknown seq_mask keys: {list(seq_mask.keys())}")

            B, L = seq_mask.shape
            x_emb = x_emb[:, :L, :]

        # Create a copy of the mask to avoid in-place operations
        seq_mask = seq_mask.clone()
        seq_mask[torch.isnan(seq_mask)] = 0

        emb = self.layer_norm(x_emb)
        emb = self.linear(emb)  # (B, L, 1)
        assert not torch.isnan(emb).any()

        # Global sum pooling for SASA (single value per structure)
        emb_masked = emb * expand(seq_mask, emb)  # Apply mask
        pooled = emb_masked.sum(dim=1)  # (B, 1) - sum instead of average

        return pooled.squeeze(-1)  # (B,)
