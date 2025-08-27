import numpy as np
import torch


def _create_special_tokens_mask(input_ids: torch.Tensor, special_token_ids: list[int]) -> torch.Tensor:
    special_tokens_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    for token in special_token_ids:
        special_tokens_mask |= input_ids == token
    special_tokens_mask = special_tokens_mask.bool()
    return special_tokens_mask


def _validate_input_shapes(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be a 2D tensor (batch_size, seq_len)")
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be a 2D tensor (batch_size, seq_len)")


def _validate_probabilities(mask_probability: float):
    if not (0 <= mask_probability <= 1):
        raise ValueError("mask_probability must be between 0 and 1")


def prepare_mlm_batch(
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

    batch_size, seq_len = input_ids.shape
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

    # Prepare output batch
    batch = {
        "input_ids": masked_input_ids,
        "labels": labels,
    }

    if pack_sequences:
        # Pack sequences for efficient flash attention
        input_ids_list = []
        labels_list = []
        position_ids_list = []
        seqlens = [0]

        for i in range(batch_size):
            # Get actual sequence length (excluding padding)
            seq_len_actual = attention_mask[i].sum().item()

            input_ids_list.append(batch["input_ids"][i, :seq_len_actual])
            labels_list.append(batch["labels"][i, :seq_len_actual])
            position_ids_list.append(torch.arange(seq_len_actual, device=device))
            seqlens.append(seq_len_actual)

        # Concatenate all sequences
        batch.update(
            {
                "input_ids": torch.cat(input_ids_list).unsqueeze(0),  # Add batch dim for model
                "labels": torch.cat(labels_list).unsqueeze(0),
                "position_ids": torch.cat(position_ids_list),
                "cu_seqlens": torch.tensor(np.cumsum(seqlens), dtype=torch.int32, device=device),
                "max_seqlen": max(seqlens[1:]),  # Exclude the initial 0
            }
        )
    else:
        # Standard padded batch format
        batch["attention_mask"] = attention_mask.bool()

    return batch
