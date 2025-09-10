import torch

from lobster.model.latent_generator.quantizer._slq import SimpleLinearQuantizer


class LigandTokenizer(torch.nn.Module):
    def __init__(
        self,
        n_tokens: int = 256,
        embed_dim: int = 4,
        softmax: bool = False,
        emb_noise: float = 0.0,
        gumbel: bool = True,
        use_gumbel_noise: bool = True,
        tau: float = 0.5,
        ligand_n_tokens: int = 256,
        ligand_embed_dim: int = 4,
        ligand_softmax: bool = False,
        ligand_emb_noise: float = 0.0,
        ligand_gumbel: bool = True,
        ligand_use_gumbel_noise: bool = True,
        ligand_tau: float = 0.5,
    ):
        super().__init__()
        self.protein_tokenizer = SimpleLinearQuantizer(
            n_tokens, embed_dim, softmax, emb_noise, gumbel, use_gumbel_noise, tau
        )
        self.ligand_tokenizer = SimpleLinearQuantizer(
            ligand_n_tokens,
            ligand_embed_dim,
            ligand_softmax,
            ligand_emb_noise,
            ligand_gumbel,
            ligand_use_gumbel_noise,
            ligand_tau,
        )

    def quantize(self, z, mask=None, ligand_mask=None):
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
