import torch.nn as nn
from torch import Tensor


class SymmetricCrossAttentionLayer(nn.Module):
    """
    Each tensor first processes itself via self-attention, then both tensors
    attend to each other via cross-attention, followed by feed-forward processing.

    Parameters
    ----------
    d_model : int
        Hidden dimension size
    nhead : int, default=8
        Number of attention heads
    dim_feedforward : int, optional
        Feed-forward network dimension. If None, defaults to 4 * d_model
    dropout : float, default=0.1
        Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        batch_first = True
        # Self-attention for each tensor
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Cross-attention between tensors
        self.cross_attn1_to_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn2_to_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm_self_attn1 = nn.LayerNorm(d_model)
        self.norm_self_attn2 = nn.LayerNorm(d_model)
        self.norm_cross_attn1 = nn.LayerNorm(d_model)
        self.norm_cross_attn2 = nn.LayerNorm(d_model)
        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.norm_ffn2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        x1_attention_mask: Tensor | None = None,
        x2_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of symmetric cross attention.

        Parameters
        ----------
        x1 : Tensor
            First sequence embeddings, shape (batch, seq_len1, d_model)
        x2 : Tensor
            Second sequence embeddings, shape (batch, seq_len2, d_model)
        x1_attention_mask : Tensor, optional
            Attention mask for x1
        x2_attention_mask : Tensor, optional
            Attention mask for x2

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated representations for x1 and x2
        """
        # self-attention
        x1_self_attended, _ = self.self_attn1(x1, x1, x1, key_padding_mask=x1_attention_mask)
        x1 = self.norm_self_attn1(x1 + self.dropout(x1_self_attended))

        x2_self_attended, _ = self.self_attn2(x2, x2, x2, key_padding_mask=x2_attention_mask)
        x2 = self.norm_self_attn2(x2 + self.dropout(x2_self_attended))

        # cross-attention
        x1_cross_attended, _ = self.cross_attn1_to_2(x1, x2, x2, key_padding_mask=x2_attention_mask)
        x1 = self.norm_cross_attn1(x1 + self.dropout(x1_cross_attended))

        x2_cross_attended, _ = self.cross_attn2_to_1(x2, x1, x1, key_padding_mask=x1_attention_mask)
        x2 = self.norm_cross_attn2(x2 + self.dropout(x2_cross_attended))

        # feed-forward
        x1_feedforward = self.ffn1(x1)
        x1 = self.norm_ffn1(x1 + self.dropout(x1_feedforward))

        x2_feedforward = self.ffn2(x2)
        x2 = self.norm_ffn2(x2 + self.dropout(x2_feedforward))

        return x1, x2


class SymmetricCrossAttentionModule(nn.Module):
    """
    Multi-layer symmetric cross attention module for molecular interactions.

    Parameters
    ----------
    d_model : int
        Hidden dimension size
    nhead : int, default=8
        Number of attention heads
    num_layers : int, default=2
        Number of symmetric cross attention layers
    dim_feedforward : int, optional
        Feed-forward network dimension. If None, defaults to 4 * d_model
    dropout : float, default=0.1
        Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                SymmetricCrossAttentionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        x1_attention_mask: Tensor | None = None,
        x2_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x1 : Tensor
            First sequence embeddings, shape (batch, seq_len1, d_model)
        x2 : Tensor
            Second sequence embeddings, shape (batch, seq_len2, d_model)
        x1_attention_mask : Tensor, optional
            Attention mask for x1
        x2_attention_mask : Tensor, optional
            Attention mask for x2

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated representations for x1 and x2
        """
        for layer in self.layers:
            x1, x2 = layer(x1, x2, x1_attention_mask, x2_attention_mask)

        return x1, x2
