"""
Diffusion Loss for continuous structure token modeling.

Adapted from MAR (Masked Autoregressive Models) paper:
"Autoregressive Image Generation without Vector Quantization"
https://arxiv.org/abs/2406.11838
https://github.com/LTH14/mar

This module provides a self-contained implementation of Diffusion Loss
that can replace categorical cross-entropy for continuous token spaces.
"""

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    From MAR/DiT: uses sinusoidal embeddings followed by MLP.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.

        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block with AdaLN modulation.

    From MAR: uses shift, scale, and gate modulation.
    :param channels: the number of input channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer with AdaLN modulation.

    From MAR/DiT: applies final normalization and linear projection.
    """

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP denoiser for Diffusion Loss.

    From MAR: Simple MLP with AdaLN conditioning.

    :param in_channels: channels in the input Tensor (target dim).
    :param model_channels: base channel count for the model (width).
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition from transformer.
    :param num_res_blocks: number of residual blocks (depth).
    :param grad_checkpointing: whether to use gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        grad_checkpointing: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following MAR/DiT conventions."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        """
        Apply the model to an input batch.

        :param x: an [N x C] or [N x L x C] Tensor of noisy inputs.
        :param t: a 1-D batch of timesteps [N].
        :param c: conditioning from transformer [N x C] or [N x L x C].
        :return: an [N x C] or [N x L x C] Tensor of outputs.
        """
        # Handle both 2D (per-sample) and 3D (per-token) inputs
        has_seq_dim = x.dim() == 3
        if has_seq_dim:
            B, L, C = x.shape
            # Flatten to [B*L, C] for processing
            x = x.reshape(B * L, C)
            c = c.reshape(B * L, -1)
            # Expand timesteps to match
            t = t.unsqueeze(1).expand(-1, L).reshape(B * L)

        x = self.input_proj(x)
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)

        y = t_emb + c_emb

        if self.grad_checkpointing and self.training:
            for block in self.res_blocks:
                x = checkpoint(block, x, y, use_reentrant=False)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        out = self.final_layer(x, y)

        # Reshape back to 3D if needed
        if has_seq_dim:
            out = out.reshape(B, L, -1)

        return out

    def forward_with_cfg(self, x: Tensor, t: Tensor, c: Tensor, cfg_scale: float) -> Tensor:
        """Forward with classifier-free guidance."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiffusionLoss(nn.Module):
    """
    Diffusion Loss for continuous structure tokens.

    Models per-token probability p(z|c) using diffusion, eliminating
    the need for vector quantization. This is a self-contained implementation
    adapted from MAR (https://github.com/LTH14/mar).

    Parameters
    ----------
    target_channels : int
        Dimension of target continuous embeddings (e.g., 256).
    z_channels : int
        Dimension of conditioning from transformer.
    depth : int
        Number of residual blocks in the MLP denoiser.
    width : int
        Hidden dimension of the MLP denoiser.
    num_sampling_steps : str
        Number of steps for sampling (e.g., "100" or "250").
    diffusion_steps : int
        Total diffusion timesteps for training.
    noise_schedule : str
        Type of noise schedule: "linear" or "cosine".
    learn_sigma : bool
        Whether to learn the variance (doubles output channels).
    grad_checkpointing : bool
        Whether to use gradient checkpointing for memory efficiency.
    """

    def __init__(
        self,
        target_channels: int,
        z_channels: int,
        depth: int = 3,
        width: int = 1024,
        num_sampling_steps: str = "100",
        diffusion_steps: int = 1000,
        noise_schedule: Literal["linear", "cosine"] = "cosine",
        learn_sigma: bool = True,
        grad_checkpointing: bool = False,
    ):
        super().__init__()

        self.target_channels = target_channels
        self.diffusion_steps = diffusion_steps
        self.learn_sigma = learn_sigma

        # Output channels: double if learning sigma (for mean + variance)
        out_channels = target_channels * 2 if learn_sigma else target_channels

        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=out_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

        # Precompute noise schedule
        betas = self._get_beta_schedule(noise_schedule, diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # For sampling
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        )
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Parse sampling steps
        self.num_sampling_steps = int(num_sampling_steps) if num_sampling_steps else diffusion_steps

    def _get_beta_schedule(self, schedule: str, num_timesteps: int) -> Tensor:
        """Create beta schedule for noise."""
        if schedule == "linear":
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """
        Forward diffusion: add noise to x_start at timestep t.

        q(x_t | x_0) = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def _extract(self, arr: Tensor, timesteps: Tensor, broadcast_shape: tuple) -> Tensor:
        """Extract values from arr at timesteps and broadcast to shape."""
        res = arr[timesteps]
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def forward(
        self,
        target: Tensor,
        z: Tensor,
        mask: Tensor | None = None,
        return_pred: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Compute diffusion loss for structure tokens.

        Parameters
        ----------
        target : Tensor [B, L, D] or [B, D]
            Ground truth continuous structure embeddings from encoder.
        z : Tensor [B, L, D] or [B, D]
            Conditioning from transformer (predicted token features).
        mask : Tensor [B, L] or [B], optional
            Valid token mask.
        return_pred : bool
            If True, also return the predicted (denoised) embeddings.

        Returns
        -------
        loss : Tensor
            Scalar diffusion loss.
        pred_x0 : Tensor [B, L, D] or [B, D], optional
            Predicted denoised embeddings (only if return_pred=True).
        """
        # Sample random timesteps
        t = torch.randint(0, self.diffusion_steps, (target.shape[0],), device=target.device)

        # Sample noise
        noise = torch.randn_like(target)

        # Forward diffusion: x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε
        x_t = self.q_sample(target, t, noise=noise)

        # Predict noise (and optionally variance)
        model_output = self.net(x_t, t, z)

        if self.learn_sigma:
            # Split output into noise prediction and variance prediction
            pred_noise, pred_var = model_output.chunk(2, dim=-1)
        else:
            pred_noise = model_output

        # MSE loss on noise prediction
        loss = (pred_noise - noise) ** 2

        # Average over feature dimension
        loss = loss.mean(dim=-1)  # [B, L] or [B]

        # Apply mask if provided
        if mask is not None:
            if loss.dim() == 2:  # [B, L]
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            else:  # [B]
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        if return_pred:
            # Predict x_0 from noise prediction (same formula as sample()):
            # x_0 = sqrt(1/α̅_t) * x_t - sqrt(1/α̅_t - 1) * pred_noise
            pred_x0 = (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise
            )
            return loss, pred_x0

        return loss

    @torch.no_grad()
    def sample(
        self,
        z: Tensor,
        temperature: float = 1.0,
        num_steps: int | None = None,
        cfg_scale: float = 1.0,
    ) -> Tensor:
        """
        Sample continuous embeddings via reverse diffusion (DDPM).

        Parameters
        ----------
        z : Tensor [B, L, D] or [B, D]
            Conditioning from transformer.
        temperature : float
            Sampling temperature (scales initial noise).
        num_steps : int, optional
            Number of sampling steps (defaults to num_sampling_steps).
        cfg_scale : float
            Classifier-free guidance scale (1.0 = no guidance).

        Returns
        -------
        x : Tensor [B, L, D] or [B, D]
            Sampled continuous embeddings.
        """
        num_steps = num_steps or self.num_sampling_steps

        # Start from pure noise
        shape = (*z.shape[:-1], self.target_channels)
        x = torch.randn(shape, device=z.device) * temperature

        # Compute timestep indices for sampling
        step_indices = torch.linspace(self.diffusion_steps - 1, 0, num_steps, device=z.device).long()

        for i, t in enumerate(step_indices):
            t_batch = torch.full((z.shape[0],), t, device=z.device, dtype=torch.long)

            # Predict noise
            if cfg_scale != 1.0:
                model_output = self.net.forward_with_cfg(x, t_batch, z, cfg_scale)
            else:
                model_output = self.net(x, t_batch, z)

            if self.learn_sigma:
                pred_noise, pred_var = model_output.chunk(2, dim=-1)
            else:
                pred_noise = model_output

            # DDPM update step (posterior uses precomputed coefs)

            # Predict x_0 from noise prediction
            pred_x0 = (
                self._extract(self.sqrt_recip_alphas_cumprod, t_batch, x.shape) * x
                - self._extract(self.sqrt_recipm1_alphas_cumprod, t_batch, x.shape) * pred_noise
            )

            # Compute posterior mean
            posterior_mean = (
                self._extract(self.posterior_mean_coef1, t_batch, x.shape) * pred_x0
                + self._extract(self.posterior_mean_coef2, t_batch, x.shape) * x
            )

            # Add noise (except at t=0)
            if t > 0:
                noise = torch.randn_like(x)
                posterior_var = self._extract(self.posterior_variance, t_batch, x.shape)
                x = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                x = posterior_mean

        return x


# Convenience function to match MAR API
def create_diffusion_loss(
    target_channels: int,
    z_channels: int,
    depth: int = 3,
    width: int = 1024,
    num_sampling_steps: str = "100",
    noise_schedule: str = "cosine",
    grad_checkpointing: bool = False,
) -> DiffusionLoss:
    """
    Factory function to create DiffusionLoss with MAR-like defaults.

    Parameters
    ----------
    target_channels : int
        Dimension of target embeddings (your structure token dim, e.g., 256).
    z_channels : int
        Dimension of conditioning (transformer hidden dim).
    depth : int
        Number of MLP residual blocks (default: 3).
    width : int
        MLP hidden dimension (default: 1024).
    num_sampling_steps : str
        Steps for inference sampling (default: "100").
    noise_schedule : str
        "linear" or "cosine" (default: "cosine").
    grad_checkpointing : bool
        Enable gradient checkpointing for memory efficiency.

    Returns
    -------
    DiffusionLoss
        Configured diffusion loss module.
    """
    return DiffusionLoss(
        target_channels=target_channels,
        z_channels=z_channels,
        depth=depth,
        width=width,
        num_sampling_steps=num_sampling_steps,
        diffusion_steps=1000,
        noise_schedule=noise_schedule,
        learn_sigma=True,
        grad_checkpointing=grad_checkpointing,
    )
