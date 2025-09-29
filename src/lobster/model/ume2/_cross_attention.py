import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from torch import Tensor

from lobster.model.neobert._rotary import apply_rotary_emb
from lobster.model.neobert._swiglu import SwiGLU


class CrossAttentionEncoderBlock(nn.Module):
    """
    Encoder block-style cross-attention layer without Flash Attention.

    Uses the same efficient components as NeoBERT (RMSNorm, SwiGLU, RoPE, SDPA)
    but adapted for cross-attention between two sequences.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension size
    num_attention_heads : int
        Number of attention heads
    intermediate_size : int
        Feed-forward network intermediate dimension
    norm_eps : float, default=1e-5
        RMS normalization epsilon
    dropout : float, default=0.0
        Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dim_head = hidden_size // num_attention_heads

        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        )

        # Cross-attention projections: Q from one sequence, K,V from another
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

        # Feed-forward network (same as NeoBERT)
        multiple_of = 8
        intermediate_size = int(2 * intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.ffn = SwiGLU(hidden_size, intermediate_size, hidden_size, bias=False)

        # Layer norms (same as NeoBERT)
        self.attention_norm_q = nn.RMSNorm(hidden_size, norm_eps)
        self.attention_norm_kv = nn.RMSNorm(hidden_size, norm_eps)
        self.ffn_norm = nn.RMSNorm(hidden_size, norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        query_seq: Tensor,
        key_value_seq: Tensor,
        query_attention_mask: Tensor | None = None,
        kv_attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass of cross-attention encoder block.

        Parameters
        ----------
        query_seq : Tensor
            Query sequence embeddings, shape (batch, seq_len_q, hidden_size)
        key_value_seq : Tensor
            Key-value sequence embeddings, shape (batch, seq_len_kv, hidden_size)
        query_attention_mask : Tensor, optional
            Attention mask for query sequence
        kv_attention_mask : Tensor, optional
            Attention mask for key-value sequence
        freqs_cis : Tensor, optional
            RoPE frequency tensor
        output_attentions : bool, default=False
            Whether to return attention weights

        Returns
        -------
        tuple[Tensor, Tensor | None]
            Updated query sequence and optional attention weights
        """
        # Cross-attention
        attn_output, attn_weights = self._cross_att_block(
            self.attention_norm_q(query_seq),
            self.attention_norm_kv(key_value_seq),
            query_attention_mask,
            kv_attention_mask,
            freqs_cis,
            output_attentions,
        )

        # Residual connection
        query_seq = query_seq + self.dropout(attn_output)

        # Feed-forward
        ffn_output = self.ffn(self.ffn_norm(query_seq))
        query_seq = query_seq + self.dropout(ffn_output)

        return query_seq, attn_weights

    def _cross_att_block(
        self,
        query_seq: Tensor,
        key_value_seq: Tensor,
        query_attention_mask: Tensor | None = None,
        kv_attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Cross-attention computation."""
        batch_size, seq_len_q, _ = query_seq.shape
        _, seq_len_kv, _ = key_value_seq.shape

        # Project to Q, K, V
        xq = self.q_proj(query_seq).view(batch_size, seq_len_q, self.num_attention_heads, self.dim_head)
        xk, xv = (
            self.kv_proj(key_value_seq)
            .view(batch_size, seq_len_kv, self.num_attention_heads, self.dim_head * 2)
            .chunk(2, dim=-1)
        )

        # Apply rotary embeddings
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Create combined attention mask for cross-attention
        attn_mask = None
        if query_attention_mask is not None or kv_attention_mask is not None:
            # create cross-attention mask: (batch, seq_len_q, seq_len_kv)
            if query_attention_mask is None:
                query_attention_mask = torch.ones(
                    batch_size, seq_len_q, device=query_seq.device, dtype=torch.bool
                ).bool()
            else:
                query_attention_mask = query_attention_mask.bool()

            if kv_attention_mask is None:
                kv_attention_mask = torch.ones(
                    batch_size, seq_len_kv, device=key_value_seq.device, dtype=torch.bool
                ).bool()
            else:
                kv_attention_mask = kv_attention_mask.bool()

            # cross-attention mask: query can attend to all valid kv positions
            attn_mask = query_attention_mask.unsqueeze(-1) & kv_attention_mask.unsqueeze(-2)

            # similar to NeoBERT: Expand for multiple heads: (batch, num_heads, seq_len_q, seq_len_kv)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)

        attn_weights = None

        # Use eager attention if attention weights are needed
        if output_attentions:
            attn_scores = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (self.dim_head**0.5)

            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

            attn_weights = attn_scores.softmax(dim=-1)
            attn = attn_weights @ xv.permute(0, 2, 1, 3)
            attn = attn.transpose(1, 2)
        else:
            # Use SDPA
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attn_mask.bool(),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        attn_output = self.wo(attn.reshape(batch_size, seq_len_q, self.hidden_size))

        return attn_output, attn_weights


class CrossAttentionModule(nn.Module):
    """
    Multi-layer cross-attention module using encoder block-style layers.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension size
    num_attention_heads : int, default=8
        Number of attention heads
    num_layers : int, default=2
        Number of cross-attention layers
    intermediate_size : int, optional
        Feed-forward network dimension. If None, defaults to 4 * hidden_size
    norm_eps : float, default=1e-5
        RMS normalization epsilon
    dropout : float, default=0.0
        Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        intermediate_size: int | None = None,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        self.layers = nn.ModuleList(
            [
                CrossAttentionEncoderBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query_seq: Tensor,
        key_value_seq: Tensor,
        query_attention_mask: Tensor | None = None,
        kv_attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """
        Forward pass through multiple cross-attention layers.

        Parameters
        ----------
        query_seq : Tensor
            Query sequence embeddings, shape (batch, seq_len_q, hidden_size)
        key_value_seq : Tensor
            Key-value sequence embeddings, shape (batch, seq_len_kv, hidden_size)
        query_attention_mask : Tensor, optional
            Attention mask for query sequence
        kv_attention_mask : Tensor, optional
            Attention mask for key-value sequence
        freqs_cis : Tensor, optional
            RoPE frequency tensor
        output_attentions : bool, default=False
            Whether to return attention weights from all layers

        Returns
        -------
        tuple[Tensor, list[Tensor] | None]
            Updated query sequence and optional attention weights from all layers
        """
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            query_seq, attn_weights = layer(
                query_seq=query_seq,
                key_value_seq=key_value_seq,
                query_attention_mask=query_attention_mask,
                kv_attention_mask=kv_attention_mask,
                freqs_cis=freqs_cis,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions.append(attn_weights)

        return query_seq, all_attentions


class SymmetricCrossAttentionModule(nn.Module):
    """
    Symmetric cross-attention using encoder block-style layers.

    Each sequence attends to the other, similar to the original implementation
    but using efficient encoder block components.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension size
    num_attention_heads : int, default=8
        Number of attention heads
    num_layers : int, default=2
        Number of symmetric cross-attention layers
    intermediate_size : int, optional
        Feed-forward network dimension. If None, defaults to 4 * hidden_size
    norm_eps : float, default=1e-5
        RMS normalization epsilon
    dropout : float, default=0.0
        Dropout probability
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        num_layers: int = 2,
        intermediate_size: int | None = None,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Cross-attention from x1 to x2
        self.cross_attn_1_to_2 = CrossAttentionModule(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            norm_eps=norm_eps,
            dropout=dropout,
        )

        # Cross-attention from x2 to x1
        self.cross_attn_2_to_1 = CrossAttentionModule(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            norm_eps=norm_eps,
            dropout=dropout,
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        x1_attention_mask: Tensor | None = None,
        x2_attention_mask: Tensor | None = None,
        freqs_cis: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Symmetric cross-attention forward pass.

        Parameters
        ----------
        x1 : Tensor
            First sequence embeddings, shape (batch, seq_len1, hidden_size)
        x2 : Tensor
            Second sequence embeddings, shape (batch, seq_len2, hidden_size)
        x1_attention_mask : Tensor, optional
            Attention mask for x1
        x2_attention_mask : Tensor, optional
            Attention mask for x2
        freqs_cis : Tensor, optional
            RoPE frequency tensor
        output_attentions : bool, default=False
            Whether to return attention weights

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated representations for x1 and x2
        """
        # x1 attends to x2
        x1_updated, _ = self.cross_attn_1_to_2(
            query_seq=x1,
            key_value_seq=x2,
            query_attention_mask=x1_attention_mask,
            kv_attention_mask=x2_attention_mask,
            freqs_cis=freqs_cis,
            output_attentions=output_attentions,
        )

        # x2 attends to x1_updated
        x2_updated, _ = self.cross_attn_2_to_1(
            query_seq=x2,
            key_value_seq=x1_updated,
            query_attention_mask=x2_attention_mask,
            kv_attention_mask=x1_attention_mask,
            freqs_cis=freqs_cis,
            output_attentions=output_attentions,
        )

        return x1_updated, x2_updated
