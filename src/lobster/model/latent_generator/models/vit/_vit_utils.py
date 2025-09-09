"""
Adapted from:
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Neural network modules. Many of these are adapted from open source modules.
"""

from typing import List, Sequence, Optional

from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from rotary_embedding_torch import RotaryEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from torch import einsum
from loguru import logger
import os
import time
from lobster.model.latent_generator.utils.residue_constants import ELEMENT_VOCAB
#os.environ["HYDRA_FULL_ERROR"] = "1"

def expand(x, tgt=None, dim=1):
    if tgt is None:
        for _ in range(dim):
            x = x[..., None]
    else:
        while len(x.shape) < len(tgt.shape):
            x = x[..., None]
    return x


########################################
# Adapted from https://github.com/ermongroup/ddim


def downsample(x):
    return nn.functional.avg_pool2d(x, 2, 2, ceil_mode=True)


def upsample_coords(x, shape):
    new_l, new_w = shape
    return nn.functional.interpolate(x, size=(new_l, new_w), mode="nearest")


########################################
# Adapted from https://github.com/aqlaboratory/openfold


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.contiguous().permute(first_inds + [zero_index + i for i in inds])



class RelativePositionalEncoding(nn.Module):
    def __init__(self, attn_dim=8, max_rel_idx=32):
        super().__init__()
        self.max_rel_idx = max_rel_idx
        self.n_rel_pos = 2 * self.max_rel_idx + 1
        self.linear = nn.Linear(self.n_rel_pos, attn_dim)

    def forward(self, residue_index):
        d_ij = residue_index[..., None] - residue_index[..., None, :]
        v_bins = torch.arange(self.n_rel_pos).to(d_ij.device) - self.max_rel_idx
        idxs = (d_ij[..., None] - v_bins[None, None]).abs().argmin(-1)
        p_ij = nn.functional.one_hot(idxs, num_classes=self.n_rel_pos)
        embeddings = self.linear(p_ij.float())
        return embeddings


########################################
# Adapted from https://github.com/NVlabs/edm


class Noise_Embedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


########################################
# Adapted from github.com/lucidrains
# https://github.com/lucidrains/denoising-diffusion-pytorch
# https://github.com/lucidrains/recurrent-interface-network-pytorch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def posemb_sincos_1d(patches, temperature=10000, residue_index=None):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device=device) if residue_index is None else residue_index
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[..., None] * omega
    pe = torch.cat((n.sin(), n.cos()), dim=-1)
    return pe.type(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class NoiseConditioningBlock(nn.Module):
    def __init__(self, n_in_channel, n_out_channel):
        super().__init__()
        self.block = nn.Sequential(
            Noise_Embedding(n_in_channel),
            nn.Linear(n_in_channel, n_out_channel),
            nn.SiLU(),
            nn.Linear(n_out_channel, n_out_channel),
            Rearrange("b d -> b 1 d"),
        )

    def forward(self, noise_level):
        return self.block(noise_level)


class TimeCondResnetBlock(nn.Module):
    def __init__(
        self, nic, noc, cond_nc, conv_layer=nn.Conv2d, dropout=0.1, n_norm_in_groups=4
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=nic // n_norm_in_groups, num_channels=nic),
            nn.SiLU(),
            conv_layer(nic, noc, 3, 1, 1),
        )
        self.cond_proj = nn.Linear(cond_nc, noc * 2)
        self.mid_norm = nn.GroupNorm(num_groups=noc // 4, num_channels=noc)
        self.dropout = dropout if dropout is None else nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=noc // 4, num_channels=noc),
            nn.SiLU(),
            conv_layer(noc, noc, 3, 1, 1),
        )
        self.mismatch = False
        if nic != noc:
            self.mismatch = True
            self.conv_match = conv_layer(nic, noc, 1, 1, 0)

    def forward(self, x, time=None):
        h = self.block1(x)

        if time is not None:
            h = self.mid_norm(h)
            scale, shift = self.cond_proj(time).chunk(2, dim=-1)
            h = (h * (expand(scale, h) + 1)) + expand(shift, h)

        if self.dropout is not None:
            h = self.dropout(h)

        h = self.block2(h)

        if self.mismatch:
            x = self.conv_match(x)

        return x + h


class TimeCondAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        heads=4,
        dim_head=32,
        norm=False,
        norm_context=False,
        time_cond_dim=None,
        attn_bias_dim=None,
        rotary_embedding_module=None,
        dropout=0.1,
        attention_dropout=0.1,
        use_sequential_to_out=False,
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim * 2))

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_bias_proj = None
        if attn_bias_dim is not None:
            self.attn_bias_proj = nn.Sequential(
                Rearrange("b a i j -> b i j a"),
                nn.Linear(attn_bias_dim, heads),
                Rearrange("b i j a -> b a i j"),
            )

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias=False)
        
        # Handle both sequential and single layer architectures
        if use_sequential_to_out:
            self.to_out = nn.Sequential(
                nn.Linear(hidden_dim, dim, bias=False),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            nn.init.zeros_(self.to_out[0].weight)
            logger.info(f"Using sequential to_out: {use_sequential_to_out}")
            logger.info(f"Dropout: {dropout}")
        else:
            self.to_out = nn.Linear(hidden_dim, dim, bias=False)
            nn.init.zeros_(self.to_out.weight)
        

        self.use_rope = False
        if rotary_embedding_module is not None:
            self.use_rope = True
            self.rope = rotary_embedding_module

        self.attention_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()
        logger.info(f"Attention dropout: {attention_dropout}")

    def forward(self, x, context=None, time=None, attn_bias=None, seq_mask=None, attn_drop_out_rate=0.0, spatial_attention_mask= None):
        # attn_bias is b, c, i, j
        h = self.heads
        has_context = exists(context)

        context = default(context, x)

        if x.shape[-1] != self.norm.gamma.shape[-1]:
            print(context.shape, x.shape, self.norm.gamma.shape)

        x = self.norm(x)

        if exists(time):
            scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if has_context:
            context = self.norm_context(context)

        if seq_mask is not None:
            x = x * seq_mask[..., None]

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        q = q * self.scale
        if self.use_rope:
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        if attn_bias is not None:
            if self.attn_bias_proj is not None:
                attn_bias = self.attn_bias_proj(attn_bias)
            sim += attn_bias

        if seq_mask is not None:
            attn_mask = torch.einsum("b i, b j -> b i j", seq_mask, seq_mask)[:, None]
            sim -= (1 - attn_mask) * 1e6

        if attn_drop_out_rate > 0.0:
            b, n = seq_mask.shape
            attn_drop_out_mask = torch.rand([b, 1, n, n], device=seq_mask.device) < attn_drop_out_rate
            attn_drop_out_mask = 1e6 * -1 * attn_drop_out_mask
            sim += attn_drop_out_mask

        if spatial_attention_mask is not None:
            spatial_attention_mask = spatial_attention_mask[:, None]
            spatial_attention_mask = 1e6 * -1 * (1 - spatial_attention_mask)
            sim += spatial_attention_mask
            
        attn = sim.softmax(dim=-1)
        attn = self.attention_dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if seq_mask is not None:
            out = out * seq_mask[..., None]
        return out


class TimeCondFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dim_out=None, time_cond_dim=None, dropout=0.1):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.norm = LayerNorm(dim)

        self.time_cond = None
        self.dropout = None
        inner_dim = int(dim * mult)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, inner_dim * 2),
            )

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        self.linear_in = nn.Linear(dim, inner_dim)
        self.nonlinearity = nn.SiLU()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        logger.info(f"Dropout: {dropout}")
        
        self.linear_out = nn.Linear(inner_dim, dim_out)
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x, time=None):
        x = self.norm(x)
        x = self.linear_in(x)
        x = self.nonlinearity(x)

        if exists(time):
            scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if exists(self.dropout):
            x = self.dropout(x)

        return self.linear_out(x)


class TimeCondTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        time_cond_dim,
        attn_bias_dim=None,
        mlp_inner_dim_mult=4,
        position_embedding_type: str = "rotary",
        dropout=0.1,
        attention_dropout=0.1,
        use_sequential_to_out=False,
    ):
        super().__init__()

        self.rope = None
        self.pos_emb_type = position_embedding_type
        if position_embedding_type == "rotary":
            if dim_head < 32:
                self.rope = RotaryEmbedding(dim=dim_head)
            else:
                self.rope = RotaryEmbedding(dim=32)
        elif position_embedding_type == "relative":
            self.relpos = nn.Sequential(
                RelativePositionalEncoding(attn_dim=heads),
                Rearrange("b i j d -> b d i j"),
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TimeCondAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            norm=True,
                            time_cond_dim=time_cond_dim,
                            attn_bias_dim=attn_bias_dim,
                            rotary_embedding_module=self.rope,
                            dropout=dropout,
                            attention_dropout=attention_dropout,
                            use_sequential_to_out=use_sequential_to_out,
                        ),
                        TimeCondFeedForward(
                            dim, mlp_inner_dim_mult, time_cond_dim=time_cond_dim, dropout=dropout
                        ),
                    ]
                )
            )

    def forward(
        self,
        x,
        time=None,
        attn_bias=None,
        context=None,
        seq_mask=None,
        residue_index=None,
        attn_drop_out_rate=0.0,
        spatial_attention_mask=None,
    ):
        if self.pos_emb_type == "absolute":
            pos_emb = posemb_sincos_1d(x)
            x = x + pos_emb
        elif self.pos_emb_type == "absolute_residx":
            assert residue_index is not None
            pos_emb = posemb_sincos_1d(x, residue_index=residue_index)
            x = x + pos_emb
        elif self.pos_emb_type == "relative":
            assert residue_index is not None
            pos_emb = self.relpos(residue_index)
            attn_bias = pos_emb if attn_bias is None else attn_bias + pos_emb
        if seq_mask is not None:
            x = x * seq_mask[..., None]

        for i, (attn, ff) in enumerate(self.layers):
            x = x + attn(
                x, context=context, time=time, attn_bias=attn_bias, seq_mask=seq_mask, attn_drop_out_rate=attn_drop_out_rate, spatial_attention_mask=spatial_attention_mask
            )
            x = x + ff(x, time=time)
            if seq_mask is not None:
                x = x * seq_mask[..., None]

        return x

class TimeCondUViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        embed_dim_hidden: int,
        patch_size: int = 1,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        n_atoms: int = 37,
        channels_per_atom: int = 6,
        attn_bias_dim: int = None,
        time_cond_dim: int = None,
        position_embedding_type: str = "rotary",
        angstrom_cutoff: float = 20,
        angstrom_cutoff_spatial: float = 20,
        pw_attn_bias: bool = False,
        concat_sine_pw: bool = False,
        spatial_attention_mask: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_sequential_to_out: bool = False,
        encode_ligand: bool = False,
        add_cls_token: bool = False,
        sequence_embedding: bool = False,
        ligand_atom_embedding: bool = False,
    ):
        super().__init__()

        # Initialize configuration params
        self.position_embedding_type = position_embedding_type
        self.n_atoms = n_atoms
        self.angstrom_cutoff = angstrom_cutoff
        self.angstrom_cutoff_spatial = angstrom_cutoff_spatial
        self.pw_attn_bias = pw_attn_bias
        self.concat_sine_pw = concat_sine_pw
        self.spatial_attention_mask = spatial_attention_mask
        self.encode_ligand = encode_ligand
        self.add_cls_token = add_cls_token
        self.sequence_embedding = sequence_embedding
        self.ligand_atom_embedding = ligand_atom_embedding
        
        if sequence_embedding:
            self.sequence_embedding = nn.Embedding(23, embed_dim_hidden)
        
        if encode_ligand: #need to make sure it is spatially aware of the protein too, especially with attention mechanism
            logger.info(f"Encoding ligand with {embed_dim_hidden} hidden dimensions")
            self.ligand_to_embedding = nn.Sequential(
                nn.Linear(3, embed_dim_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(embed_dim_hidden, embed_dim_hidden),
                LayerNorm(embed_dim_hidden),
            )
            
            # Ligand atom type embedding using element vocabulary
            if ligand_atom_embedding:
                logger.info(f"Adding ligand atom type embeddings")
                # Use element vocabulary size
                self.ligand_atom_type_embedding = nn.Embedding(len(ELEMENT_VOCAB), embed_dim_hidden)
        transformer_seq_len = seq_len // (2**1)
        assert transformer_seq_len % patch_size == 0

        patch_dim = patch_size * n_atoms * channels_per_atom

        # Make transformer
        if add_cls_token:
            logger.info(f"Adding CLS token to encoder")
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim_hidden))
            
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) a -> b n (p c a)", p=patch_size),
            nn.Linear(patch_dim, embed_dim_hidden),
            LayerNorm(embed_dim_hidden),
        )

        self.transformer = TimeCondTransformer(
            embed_dim_hidden,
            depth,
            heads,
            dim_head,
            time_cond_dim,
            attn_bias_dim=attn_bias_dim,
            position_embedding_type=position_embedding_type,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
        )

        if self.concat_sine_pw:
            logger.info(f"Using concat_sine_pw with angstrom cutoff {self.angstrom_cutoff}")
            self.pw_to_embedding = nn.Sequential(
                nn.Linear((n_atoms)**2, embed_dim_hidden, bias=False),
                nn.ReLU(),
                nn.Linear(embed_dim_hidden, embed_dim_hidden),
                LayerNorm(embed_dim_hidden),
            )
            self.concat_sine_pw_embedding = nn.Sequential(
                nn.Linear(embed_dim_hidden + embed_dim_hidden, embed_dim_hidden),
                nn.ReLU(),
                nn.Linear(embed_dim_hidden, embed_dim_hidden),
                LayerNorm(embed_dim_hidden),
            )


        self.to_hidden = nn.Sequential(
            LayerNorm(embed_dim_hidden),
            nn.Linear(embed_dim_hidden, embed_dim),
        )

    def pairwise_distances_with_ligand(self, coords, ligand_coords, clamp=True):
        B, L_ligand, _ = ligand_coords.shape
        if coords is not None:
            B, L, n_atoms, I = coords.shape
            L = L_ligand + L
        else:
            L = L_ligand

        if coords is not None:
            coords_n = coords[:, :, 0, :]
            coords_ca = coords[:, :, 1, :]
            coords_c = coords[:, :, 2, :]
            coords_n = torch.cat([coords_n, ligand_coords], dim=1) #concatenate ligand coords to protein N coords
            coords_ca = torch.cat([coords_ca, ligand_coords], dim=1) #concatenate ligand coords to protein Ca coords
            coords_c = torch.cat([coords_c, ligand_coords], dim=1) #concatenate ligand coords to protein C coords
            coords_n_ = torch.cdist(coords_n, coords_n, p=2).unsqueeze(-1)
            coords_ca_ = torch.cdist(coords_ca, coords_ca, p=2).unsqueeze(-1)
            coords_c_ = torch.cdist(coords_c, coords_c, p=2).unsqueeze(-1)
            coords_n_ca_ = torch.cdist(coords_n, coords_ca, p=2).unsqueeze(-1)
            coords_n_c_ = torch.cdist(coords_n, coords_c, p=2).unsqueeze(-1)
            coords_ca_c_ = torch.cdist(coords_ca, coords_c, p=2).unsqueeze(-1)
        else:
            coords_lig = torch.cdist(ligand_coords, ligand_coords, p=2).unsqueeze(-1)

        if clamp:
            if coords is not None:
                coords_n_ = torch.clamp(coords_n_, max=self.angstrom_cutoff)
                coords_ca_ = torch.clamp(coords_ca_, max=self.angstrom_cutoff)
                coords_c_ = torch.clamp(coords_c_, max=self.angstrom_cutoff)
                coords_n_ca_ = torch.clamp(coords_n_ca_, max=self.angstrom_cutoff)
                coords_n_c_ = torch.clamp(coords_n_c_, max=self.angstrom_cutoff)
                coords_ca_c_ = torch.clamp(coords_ca_c_, max=self.angstrom_cutoff)
            else:
                coords_lig = torch.clamp(coords_lig, max=self.angstrom_cutoff)
        #note that we are repeating some to fit the expected 9 dim embedding layer
        if coords is not None:
            coords = torch.cat([coords_n_, coords_ca_, coords_c_, coords_n_ca_, coords_n_c_, coords_ca_c_, coords_n_ca_, coords_n_c_, coords_ca_c_], dim=-1)
        else:
            coords = torch.cat([coords_lig, coords_lig, coords_lig, coords_lig, coords_lig, coords_lig, coords_lig, coords_lig, coords_lig], dim=-1)
        return coords

    def pairwise_distances(self, coords, ligand_coords=None, clamp=True):
        if ligand_coords is not None:
            return self.pairwise_distances_with_ligand(coords, ligand_coords, clamp)
        B, L, n_atoms, I = coords.shape
        coords = coords.reshape(B, -1, 3)
        # calculate pairwise distances
        coords = torch.cdist(coords, coords, p=2)
        # clamp to 20 Angstroms
        if clamp:
            coords = torch.clamp(coords, max=self.angstrom_cutoff)
        return coords.reshape(B, L, L, -1)

    def create_distance_attention_mask(self, distance_matrix, max_distance= 30.0):
        """Create an attention mask based on distance cutoff.

        Args:
            distance_matrix: Tensor of shape (B, L, L) containing pairwise distances in Angstroms
            max_distance: Float, maximum distance in Angstroms to allow attention (default: 30.0)

        Returns:
            attention_mask: Boolean tensor of shape (B, L, L) where True indicates positions
                        that should be attended to (distances <= max_distance)
        """
        # Create mask where True indicates valid attention (distance <= max_distance)
        attention_mask = distance_matrix < max_distance

        # Ensure diagonal is always attended to (self-attention)
        batch_size, seq_len, _ = distance_matrix.shape
        diagonal_mask = torch.eye(seq_len, device=distance_matrix.device).bool()
        diagonal_mask = diagonal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        attention_mask = attention_mask | diagonal_mask

        return attention_mask

    def forward(
        self, coords, time_cond=None, pair_bias=None, seq_mask=None, residue_index=None, attn_drop_out_rate=0.0, return_embeddings=False, ligand_coords=None, ligand_mask=None, ligand_residue_index=None, sequence=None, ligand_atom_types=None, **kwargs
    ):
        if coords is not None:
            B,L,n_atoms = coords.shape[:3]
            coords_gt = coords.clone()
            x = rearrange(coords, "b n a c -> b c n a")
            x = self.to_patch_embedding(x)
            if self.sequence_embedding and sequence is not None:
                x = x + self.sequence_embedding(sequence)
        else:
            coords_gt = None

        if self.encode_ligand and ligand_coords is not None:
            ligand_embedding = self.ligand_to_embedding(ligand_coords)
            
            # Add ligand atom type embeddings if available
            if self.ligand_atom_embedding and ligand_atom_types is not None:
                ligand_type_embedding = self.ligand_atom_type_embedding(ligand_atom_types)
                ligand_embedding = ligand_embedding + ligand_type_embedding
            
            if coords is not None:
                x = torch.cat([x, ligand_embedding], -2)
                seq_mask = torch.cat([seq_mask, ligand_mask], -1)
                residue_index = torch.cat([residue_index, ligand_residue_index], -1)
            else:
                x = ligand_embedding
                seq_mask = ligand_mask
                residue_index = ligand_residue_index

        if self.concat_sine_pw:
            with torch.no_grad():
                pw_coords = self.pairwise_distances(coords_gt, ligand_coords=ligand_coords)
            pw_coords = self.pw_to_embedding(pw_coords)
            pw_coords = pw_coords.mean(2).squeeze(2)
            x = self.concat_sine_pw_embedding(torch.cat([x, pw_coords], -1))

        if self.spatial_attention_mask:
            with torch.no_grad():
                pw_distance_matrix = self.pairwise_distances(coords_gt, ligand_coords=ligand_coords, clamp=False)[:, :, :, 1]
                spatial_attention_mask_ = self.create_distance_attention_mask(pw_distance_matrix, self.angstrom_cutoff_spatial).float()
        else:
            spatial_attention_mask_ = None

        if seq_mask is not None and x.shape[1] == seq_mask.shape[1]:
            x *= seq_mask[..., None]

        # Prepend CLS token
        if self.add_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            seq_mask = torch.cat([torch.ones(B, 1, device=x.device), seq_mask], dim=1)
            residue_index = torch.cat([torch.zeros(B, 1, device=x.device), residue_index], dim=1)
            if spatial_attention_mask_ is not None:
                padding_col = torch.ones(B, 1, L, device=x.device)
                padding_row = torch.ones(B, L+1, 1, device=x.device)
                spatial_attention_mask_ = torch.cat([padding_col, spatial_attention_mask_], dim=1)
                spatial_attention_mask_ = torch.cat([padding_row, spatial_attention_mask_], dim=2)

        x = self.transformer(
            x,
            time=time_cond,
            attn_bias=pair_bias,
            seq_mask=seq_mask,
            residue_index=residue_index,
            attn_drop_out_rate=attn_drop_out_rate,
            spatial_attention_mask = spatial_attention_mask_,
        )
        
        x_out = self.to_hidden(x)

        if return_embeddings:
            return x_out, x

        return x_out

class TimeCondUViTDecoder(nn.Module):
    def __init__(
        self,
        *,
        struc_token_codebook_size: int,
        struc_token_dim: int,
        seq_len: int,
        patch_size: int = 1,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        n_atoms: int = 37,
        attn_bias_dim: int = None,
        time_cond_dim: int = None,
        position_embedding_type: str = "rotary",
        indexed: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_sequential_to_out: bool = False,
        encode_ligand: bool = False,
        ligand_struc_token_codebook_size: int = 256,
        refinement_module: bool = False,
    ):
        super().__init__()

        self.position_embedding_type = position_embedding_type
        self.encode_ligand = encode_ligand
        logger.info(f"Decoder encode_ligand set to: {self.encode_ligand}")

        transformer_seq_len = seq_len // (2**1)
        assert transformer_seq_len % patch_size == 0

        patch_dim_out = patch_size * n_atoms * 3
        dim_a = n_atoms

        self.refinement_module = refinement_module


        #embedding for structure tokens
        if indexed:
            self.embed_struc_tokens = nn.Embedding(struc_token_codebook_size, struc_token_dim)
        else:
            self.embed_struc_tokens = nn.Linear(struc_token_codebook_size, struc_token_dim, bias=False)

        if self.encode_ligand:
            logger.info(f"Encoding ligand with {struc_token_dim} hidden dimensions")
            self.embed_ligand_tokens = nn.Linear(ligand_struc_token_codebook_size, struc_token_dim, bias=False)
            self.from_patch_ligand = nn.Sequential(
                    LayerNorm(struc_token_dim),
                    nn.Linear(struc_token_dim, 3),
                )


        self.transformer = TimeCondTransformer(
            struc_token_dim,
            depth,
            heads,
            dim_head,
            time_cond_dim,
            attn_bias_dim=attn_bias_dim,
            position_embedding_type=position_embedding_type,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
        )

        if self.refinement_module:
            self.refinement_module_transformer = TimeCondTransformer(
                struc_token_dim,
                depth,
                heads,
                dim_head,
                time_cond_dim,
                attn_bias_dim=attn_bias_dim,
                position_embedding_type=position_embedding_type,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_sequential_to_out=use_sequential_to_out,
                )
            self.refinement_ffn = nn.Sequential(
                LayerNorm(struc_token_dim),
                nn.Linear(struc_token_dim, struc_token_dim),
                nn.ReLU(),
                nn.Linear(struc_token_dim, struc_token_dim),
            )
            self.refinement_to_coords = nn.Sequential(
                LayerNorm(struc_token_dim),
                nn.Linear(struc_token_dim, patch_dim_out),
                Rearrange("b n (p c a) -> b c (n p) a", p=patch_size, a=dim_a),
            )

        #make nn.sequence for residual connection after transformer
        self.ffn = nn.Sequential(
            LayerNorm(struc_token_dim),
            nn.Linear(struc_token_dim, struc_token_dim),
            nn.ReLU(),
            nn.Linear(struc_token_dim, struc_token_dim),
        )

        self.from_patch = nn.Sequential(
            LayerNorm(struc_token_dim),
            nn.Linear(struc_token_dim, patch_dim_out),
            Rearrange("b n (p c a) -> b c (n p) a", p=patch_size, a=dim_a),
        )


    def forward(
        self, x_quant, time_cond=None, pair_bias=None, seq_mask=None, residue_index=None, ligand_quant=None, ligand_mask=None, **kwargs
    ):
        if x_quant is not None:
            x_emb = self.embed_struc_tokens(x_quant)
        else:
            x_emb = None
        if self.encode_ligand and ligand_quant is not None:
            ligand_emb = self.embed_ligand_tokens(ligand_quant)
            if x_emb is not None:
                B, L, D = x_emb.shape
                x_emb = torch.cat([x_emb, ligand_emb], -2)
                seq_mask = torch.cat([seq_mask, ligand_mask], -1)
            else:
                x_emb = ligand_emb
                seq_mask = ligand_mask

        if seq_mask is not None and x_emb.shape[1] == seq_mask.shape[1]:
            x_emb *= seq_mask[..., None]

        x = x_emb
        x_out = self.transformer(
            x_emb,
            time=time_cond,
            attn_bias=pair_bias,
            seq_mask=seq_mask,
            residue_index=residue_index,
        )
        #add skip connection
        x_out = x_out + x_emb

        x = self.ffn(x_out)

        if self.encode_ligand and ligand_quant is not None:
            if x_quant is not None:
                ligand_x = x[:, L:, :]
                x = x[:, :L, :]
            else:
                ligand_x = x
                x = None
            ligand_x = self.from_patch_ligand(ligand_x)

        if x_quant is not None:
            x = self.from_patch(x)
            x = rearrange(x, "b c n a -> b n a c")

        if self.encode_ligand and ligand_quant is not None:
            out = {"protein_coords": x, "ligand_coords": ligand_x}
        else:
            out = x
        
        if self.refinement_module:
            x_out = self.ffn(x_out)
            x_refinement = self.refinement_module_transformer(x_out)
            x_refinement = x_refinement + x_out
            x_refinement = self.refinement_ffn(x_refinement)
            x_refinement = self.refinement_to_coords(x_refinement)
            x_refinement = rearrange(x_refinement, "b c n a -> b n a c")
            #if dictionary, add to it
            if isinstance(out, dict):
                out["protein_coords_refinement"] = x_refinement
            else:
                out = {"protein_coords": x, "protein_coords_refinement": x_refinement}


        return out

class TimeCondUViTAlignDecoder(nn.Module):
    def __init__(
        self,
        *,
        struc_token_codebook_size: int,
        struc_token_dim: int,
        seq_len: int,
        patch_size: int = 1,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        n_atoms: int = 37,
        attn_bias_dim: int = None,
        time_cond_dim: int = None,
        position_embedding_type: str = "rotary",
        indexed: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_sequential_to_out: bool = False,
    ):
        super().__init__()

        self.position_embedding_type = position_embedding_type

        transformer_seq_len = seq_len // (2**1)
        assert transformer_seq_len % patch_size == 0

        patch_dim_out = patch_size * n_atoms * 3
        dim_a = n_atoms


        #embedding for structure tokens
        if indexed:
            self.embed_struc_tokens = nn.Embedding(struc_token_codebook_size, struc_token_dim)
        else:
            self.embed_struc_tokens = nn.Linear(struc_token_codebook_size, struc_token_dim, bias=False)


        self.transformer = TimeCondTransformer(
            struc_token_dim,
            depth,
            heads,
            dim_head,
            time_cond_dim,
            attn_bias_dim=attn_bias_dim,
            position_embedding_type=position_embedding_type,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
        )

        #make nn.sequence for residual connection after transformer
        self.ffn = nn.Sequential(
            LayerNorm(struc_token_dim),
            nn.Linear(struc_token_dim, struc_token_dim),
            nn.ReLU(),
            nn.Linear(struc_token_dim, struc_token_dim),
        )

    def forward(
        self, x_quant, time_cond=None, pair_bias=None, seq_mask=None, residue_index=None, **kwargs
    ):

        x_emb = self.embed_struc_tokens(x_quant)

        if seq_mask is not None and x_emb.shape[1] == seq_mask.shape[1]:
            x_emb *= seq_mask[..., None]

        x = x_emb
        x = self.transformer(
            x_emb,
            time=time_cond,
            attn_bias=pair_bias,
            seq_mask=seq_mask,
            residue_index=residue_index,
        )
        #add skip connection
        x = x + x_emb

        x = self.ffn(x)

        return x
    
class PLMUViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        embed_dim: int,
        embed_dim_hidden: int,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        attn_bias_dim: int = None,
        time_cond_dim: int = None,
        position_embedding_type: str = "rotary",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_sequential_to_out: bool = False,
        plm_hidden_dim: int = 960,
        use_template_coords: bool = False,
    ):
        super().__init__()

        # Initialize configuration params
        self.position_embedding_type = position_embedding_type
        self.sequence_embedding = nn.Linear(plm_hidden_dim, embed_dim_hidden, bias=False)
        self.use_template_coords = use_template_coords
        if self.use_template_coords:
            self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) a -> b n (p c a)", p=1),
            nn.Linear(9, embed_dim_hidden),
            LayerNorm(embed_dim_hidden),
            )

        self.transformer = TimeCondTransformer(
            embed_dim_hidden,
            depth,
            heads,
            dim_head,
            time_cond_dim,
            attn_bias_dim=attn_bias_dim,
            position_embedding_type=position_embedding_type,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_sequential_to_out=use_sequential_to_out,
        )



        self.to_hidden = nn.Sequential(
            LayerNorm(embed_dim_hidden),
            nn.Linear(embed_dim_hidden, embed_dim),
        )

    def forward(
        self, plm_embeddings, time_cond=None, seq_mask=None, residue_index=None, coords=None, attn_drop_out_rate=0.0, return_embeddings=False, **kwargs
    ):
        x = self.sequence_embedding(plm_embeddings)

        if self.use_template_coords:
            x_template = rearrange(coords, "b n a c -> b c n a")
            x_template = self.to_patch_embedding(x_template)
            x = x + x_template

        if seq_mask is not None and x.shape[1] == seq_mask.shape[1]:
            x *= seq_mask[..., None]


        x = self.transformer(
            x,
            time=time_cond,
            attn_bias=None,
            seq_mask=seq_mask,
            residue_index=residue_index,
            attn_drop_out_rate=attn_drop_out_rate,
            spatial_attention_mask = None,
        )
        
        x_out = self.to_hidden(x)

        if return_embeddings:
            return x_out, x

        return x_out
