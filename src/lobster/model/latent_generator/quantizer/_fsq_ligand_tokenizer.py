import torch

from lobster.model.latent_generator.quantizer._fsq import FiniteScalarQuantizer


class FSQLigandTokenizer(torch.nn.Module):
    """Ligand tokenizer using Finite Scalar Quantization (FSQ) for both protein and ligand."""

    def __init__(
        self,
        protein_levels: list[int] | None = None,
        ligand_levels: list[int] | None = None,
        return_oh_like: bool = True,
        n_tokens: int | None = None,  # For hydra config compatibility (ignored, computed from levels)
    ):
        """
        Initialize FSQ-based ligand tokenizer.

        Args:
            protein_levels: FSQ levels for protein tokenization (e.g., [8, 6, 5] for 240 tokens)
            ligand_levels: FSQ levels for ligand tokenization (e.g., [8, 6, 5] for 240 tokens)
            return_oh_like: Whether to return one-hot-like representation
            n_tokens: Ignored - for hydra config compatibility only (n_tokens is computed from levels)
        """
        del n_tokens  # Not used - computed from levels
        super().__init__()
        if protein_levels is None:
            protein_levels = [8, 6, 5]  # 240 tokens
        if ligand_levels is None:
            ligand_levels = [8, 6, 5]  # 240 tokens

        self.protein_tokenizer = FiniteScalarQuantizer(
            levels=protein_levels,
            return_oh_like=return_oh_like,
        )
        self.ligand_tokenizer = FiniteScalarQuantizer(
            levels=ligand_levels,
            return_oh_like=return_oh_like,
        )

        # Store codebook sizes for external use
        self.n_tokens = self.protein_tokenizer.n_tokens
        self.ligand_n_tokens = self.ligand_tokenizer.n_tokens

    def quantize(self, z, mask=None, ligand_mask=None):
        """
        Quantize protein and ligand embeddings using FSQ.

        Args:
            z: Input embeddings of shape (B, L_protein + L_ligand, embed_dim)
            mask: Protein mask of shape (B, L_protein)
            ligand_mask: Ligand mask of shape (B, L_ligand)

        Returns:
            out_tokens: Dict with 'protein_tokens' and 'ligand_tokens'
            out_logits: Dict with 'protein_logits' and 'ligand_logits'
            out_masks: Dict with 'protein_mask' and 'ligand_mask'
        """
        if mask is not None:
            B, L = mask.shape
            z_protein = z[:, :L, :]
            z_ligand = z[:, L:, :]
            protein_tokens, protein_logits, protein_mask = self.protein_tokenizer.quantize(z_protein, mask)
            ligand_tokens, ligand_logits, ligand_mask = self.ligand_tokenizer.quantize(z_ligand, ligand_mask)
            out_tokens = {"protein_tokens": protein_tokens, "ligand_tokens": ligand_tokens}
            out_logits = {"protein_logits": protein_logits, "ligand_logits": ligand_logits}
            out_masks = {"protein_mask": protein_mask, "ligand_mask": ligand_mask}
        else:
            ligand_tokens, ligand_logits, ligand_mask = self.ligand_tokenizer.quantize(z, ligand_mask)
            out_tokens = {"ligand_tokens": ligand_tokens}
            out_logits = {"ligand_logits": ligand_logits}
            out_masks = {"ligand_mask": ligand_mask}
        return out_tokens, out_logits, out_masks
