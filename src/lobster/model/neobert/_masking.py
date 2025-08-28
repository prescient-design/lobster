import torch


def _create_special_tokens_mask(input_ids: torch.Tensor, special_token_ids: list[int]) -> torch.Tensor:
    special_tokens_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    for token in special_token_ids:
        special_tokens_mask |= input_ids == token
    special_tokens_mask = special_tokens_mask.bool()

    return special_tokens_mask


def _validate_input_shapes(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be a 2D tensor (batch_size, seq_len) but got {input_ids.shape}")
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be a 2D tensor (batch_size, seq_len) but got {attention_mask.shape}")


def _validate_probabilities(mask_probability: float):
    if not (0 <= mask_probability <= 1):
        raise ValueError("mask_probability must be between 0 and 1")


def mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    mask_probability: float = 0.2,
    pack_sequences: bool = False,
    generator: torch.Generator | None = None,
    special_token_ids: list[int] = None,
) -> dict[str, torch.Tensor]:
    """
    Prepare masked language modeling batch with optional sequence packing.

    Parameters
    ----------
    input_ids : torch.Tensor
        Input token IDs as 2D tensor (batch_size, seq_len).
    attention_mask : torch.Tensor
        Attention mask with same shape as input_ids. 1 for real tokens, 0 for padding.
    mask_token_id : int
        Token ID for the mask token (e.g., [MASK]).
    mlm_probability : float, default=0.2
        Probability of masking each token.
    pack_sequences : bool, default=False
        Whether to pack sequences for efficient attention computation.
    generator : torch.Generator | None, default=None
        Random number generator for reproducible masking.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - input_ids: Masked input token IDs
        - labels: Labels for MLM loss (-100 for non-masked tokens)
        - attention_mask: Attention mask (if not packing)
        - position_ids: Position IDs (if packing)
        - cu_seqlens: Cumulative sequence lengths (if packing)
        - max_seqlen: Maximum sequence length (if packing)

    Notes
    -----
    - Special tokens are not masked
    - All selected tokens are replaced with [MASK] token
    - When packing, sequences are concatenated and flash attention parameters are included

    Examples
    --------
    >>> # Batch with packing
    >>> input_ids = torch.tensor([[101, 2054, 2003, 0], [101, 7592, 102, 0]])
    >>> attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    >>> batch = prepare_mlm_batch(input_ids, attention_mask, mask_token_id=103,
    ...                          pack_sequences=True)
    >>> print(batch.keys())
    dict_keys(['input_ids', 'labels', 'position_ids', 'cu_seqlens', 'max_seqlen'])
    """
    _validate_input_shapes(input_ids, attention_mask)
    _validate_probabilities(mask_probability)

    device = input_ids.device

    # Clone inputs to avoid modifying originals
    masked_input_ids = input_ids.clone()
    labels = input_ids.clone()

    # Create probability matrix for masking
    probability_matrix = torch.full(input_ids.shape, mask_probability, device=device)

    # Don't mask special tokens
    special_tokens_mask = _create_special_tokens_mask(input_ids, special_token_ids)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Sample tokens to mask
    masked_indices = torch.bernoulli(probability_matrix, generator=generator).bool()

    # Only compute loss on masked tokens
    labels[~masked_indices] = -100

    # Replace all masked tokens with [MASK]
    masked_input_ids[masked_indices] = mask_token_id

    if not pack_sequences:
        # Standard (unpacked) batch output
        return {
            "input_ids": masked_input_ids,
            "labels": labels,
            "attention_mask": attention_mask.to(torch.bool),
        }
    else:
        raise NotImplementedError("Packing is not implemented yet")
