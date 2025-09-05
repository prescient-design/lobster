import numpy as np
import torch
from icecream import ic

from lobster.model.latent_generator.latent_generator.utils import gumbel_softmax


class SimpleLinearQuantizer(torch.nn.Module):
    def __init__(
        self,
        n_tokens: int = 1000,
        embed_dim: int = 4,
        softmax: bool = False,
        emb_noise: float = 0.0,
        gumbel: bool = True,
        use_gumbel_noise: bool = True,
        tau: float = 0.5,
        **kwargs,
    ):
        super(SimpleLinearQuantizer, self).__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.softmax = softmax
        self.emb_noise = emb_noise
        self.gumbel = gumbel
        self.use_gumbel_noise = use_gumbel_noise
        self.tau = tau


        # layer norm
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim)
        self.linear = torch.nn.Linear(self.embed_dim, n_tokens, bias=False)

    def quantize(self, z, mask=None, **kwargs):
        z_norm = self.layer_norm(z)
        z_emb = self.linear(z_norm)

        if self.gumbel:
            z_tokens = gumbel_softmax(z_emb, temperature=self.tau, hard=False, include_noise=self.use_gumbel_noise)

            return z_tokens, z_emb, mask

        elif self.softmax:
            z_emb_noisy = z_emb + self.emb_noise * torch.randn_like(z_emb)
            if mask is None:
                mask = torch.ones_like(z_emb_noisy[..., 0]).to(z_emb_noisy.device)
            z_emb_noisy = z_emb_noisy  * mask.clone().unsqueeze(-1)

            z_tokens = torch.nn.functional.softmax(z_emb_noisy/self.tau, dim=-1)

            return z_tokens, z_emb, mask

        else:
            z_tokens = torch.argmax(z_emb, dim=-1)

            z_tokens = torch.nn.functional.one_hot(z_tokens, num_classes=self.n_tokens).float()

            return z_emb + (z_tokens - z_emb).detach(), z_emb, mask



