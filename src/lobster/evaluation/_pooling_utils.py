"""Shared pooling utilities for DGEB adapters."""

import torch
from typing import Literal


def apply_dgeb_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    pool_type: Literal["mean", "max", "cls", "last"] = "mean"
) -> torch.Tensor:
    """Apply pooling to token-level embeddings with proper attention masking.
    
    This function implements the standard pooling strategies used by both
    UME and ESM adapters, ensuring consistent behavior across different models.
    
    Parameters
    ----------
    token_embeddings : torch.Tensor
        Token-level embeddings of shape (batch_size, seq_len, hidden_size)
    attention_mask : torch.Tensor
        Attention mask of shape (batch_size, seq_len) where 1 indicates real tokens
        and 0 indicates padding tokens
    pool_type : str, default="mean"
        Pooling strategy. One of "mean", "max", "cls", "last"
        
    Returns
    -------
    torch.Tensor
        Pooled embeddings of shape (batch_size, hidden_size)
    """
    if pool_type == "mean":
        # Masked mean pooling - exclude padding tokens
        masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)  # (batch_size, hidden_size)
        token_counts = attention_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
        token_counts = torch.clamp(token_counts, min=1.0)  # Avoid division by zero
        pooled = sum_embeddings / token_counts
        
    elif pool_type == "max":
        # Masked max pooling - set padding positions to -inf before max
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
        masked_embeddings = token_embeddings.masked_fill(mask_expanded == 0, float("-inf"))
        pooled = masked_embeddings.max(dim=1)[0]
        
    elif pool_type == "cls":
        # Use the first token (typically CLS token)
        pooled = token_embeddings[:, 0, :]
        
    elif pool_type == "last":
        # Use the last real token (before padding)
        lengths = attention_mask.sum(dim=1) - 1  # Last real token index
        pooled = torch.stack([
            token_embeddings[i, l.long(), :] 
            for i, l in enumerate(lengths)
        ], dim=0)
        
    else:
        raise ValueError(f"Unsupported pool_type: {pool_type}")
    
    return pooled


def create_attention_mask_from_embeddings(
    token_embeddings: torch.Tensor,
    threshold: float = 1e-8
) -> torch.Tensor:
    """Create attention mask by detecting non-zero embeddings.
    
    This is useful for models like ESM that don't provide explicit attention masks
    but use zero embeddings for padding tokens.
    
    Parameters
    ----------
    token_embeddings : torch.Tensor
        Token-level embeddings of shape (batch_size, seq_len, hidden_size)
    threshold : float, default=1e-8
        Threshold for considering embeddings as non-zero
        
    Returns
    -------
    torch.Tensor
        Attention mask of shape (batch_size, seq_len)
    """
    # Compute L2 norm across embedding dimension
    emb_norm = torch.norm(token_embeddings, dim=-1)  # (batch_size, seq_len)
    
    # Create mask: 1 for real tokens, 0 for padding
    attention_mask = (emb_norm > threshold).float()
    
    return attention_mask