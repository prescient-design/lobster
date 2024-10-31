# Adapted from https://huggingface.co/McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp/blob/main/attn_mask_utils.py
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def _prepare_4d_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=False, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length
    batch_size, query_length = input_shape

    # torch.jit.trace and torchdynamo with fullgraph=True are unable to capture the controlflow `is_causal=attention_mask is None and q_len > 1`
    # used as an SDPA argument. We keep compatibility with these tracing tools by always using SDPA's `attn_mask` argument in case we are tracing.
    # TODO: Fix this as well when using torchdynamo with fullgraph=True.
    is_tracing = torch.jit.is_tracing()

    if attention_mask is not None:
        if torch.all(attention_mask == 1):
            if is_tracing:
                pass
            elif query_length == 1:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                attention_mask = None
            elif key_value_length == query_length:
                attention_mask = None
            else:
                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
                # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                pass
    elif query_length > 1 and key_value_length != query_length:
        # See the comment above (https://github.com/pytorch/pytorch/issues/108108).
        # Ugly: we set it to True here to dispatch in the following controlflow to `to_causal_4d`.
        attention_mask = True
    elif is_tracing:
        raise ValueError(
            'Attention using SDPA can not be traced with torch.jit.trace when no attention_mask is provided. To solve this issue, please either load your model with the argument `attn_implementation="eager"` or pass an attention_mask input when tracing the model.'
        )

    if attention_mask is None:
        expanded_4d_mask = None
    elif attention_mask is True:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
    else:
        expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            dtype=inputs_embeds.dtype,
            key_value_length=key_value_length,
        )

        # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
        # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
        if query_length > 1:
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, attention_mask, unmasked_value=0.0
            )

    return expanded_4d_mask


def _prepare_4d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=False, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    return attention_mask


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=False, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    return attention_mask


def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Prepares the correct `attn_mask` argument to be used by `torch.nn.functional.scaled_dot_product_attention`.
    In case no token is masked in the `attention_mask` argument, we simply set it to `None` for the cases `query_length == 1` and
    `key_value_length == query_length`, and rely instead on SDPA `is_causal` argument to use causal/non-causal masks,
    allowing to dispatch to the flash attention kernel (that can otherwise not be used if a custom `attn_mask` is passed).
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=False, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length
    batch_size, query_length = input_shape

    # torch.jit.trace, symbolic_trace and torchdynamo with fullgraph=True are unable to capture the controlflow `is_causal=attention_mask is None and q_len > 1`
    # used as an SDPA argument. We keep compatibility with these tracing tools by always using SDPA's `attn_mask` argument in case we are tracing.
    # TODO: Fix this as well when using torchdynamo with fullgraph=True.
    is_tracing = torch.jit.is_tracing() or isinstance(inputs_embeds, torch.fx.Proxy)

    if attention_mask is not None:
        # 4d mask is passed through
        if len(attention_mask.shape) == 4:
            expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
                )
            else:
                # if the 4D mask has correct shape - invert it and fill with negative infinity
                inverted_mask = 1.0 - attention_mask.to(inputs_embeds.dtype)
                attention_mask = inverted_mask.masked_fill(
                    inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
                )
                return attention_mask

        elif not is_tracing and torch.all(attention_mask == 1):
            if query_length == 1:
                # For query_length == 1, causal attention and bi-directional attention are the same.
                attention_mask = None
            elif key_value_length == query_length:
                attention_mask = None
            else:
                # Unfortunately, for query_length > 1 and key_value_length != query_length, we cannot generally ignore the attention mask, as SDPA causal mask generation
                # may be wrong. We will set `is_causal=False` in SDPA and rely on Transformers attention_mask instead, hence not setting it to None here.
                # Reference: https://github.com/pytorch/pytorch/issues/108108
                pass
    elif query_length > 1 and key_value_length != query_length:
        # See the comment above (https://github.com/pytorch/pytorch/issues/108108).
        # Ugly: we set it to True here to dispatch in the following controlflow to `to_causal_4d`.
        attention_mask = True
    elif is_tracing:
        raise ValueError(
            'Attention using SDPA can not be traced with torch.jit.trace when no attention_mask is provided. To solve this issue, please either load your model with the argument `attn_implementation="eager"` or pass an attention_mask input when tracing the model.'
        )

    if attention_mask is None:
        expanded_4d_mask = None
    elif attention_mask is True:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
    else:
        expanded_4d_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            dtype=inputs_embeds.dtype,
            key_value_length=key_value_length,
        )

        # From PyTorch 2.1 onwards, F.scaled_dot_product_attention with the memory-efficient attention backend
        # produces nans if sequences are completely unattended in the attention mask. Details: https://github.com/pytorch/pytorch/issues/110213
        #
        # This fix is not applied in case we are tracing with torch.jit.trace or symbolic_trace, as _unmask_unattended has a data-dependent
        # controlflow that can not be captured properly.
        # TODO: _unmask_unattended does not work either with torch.compile when using fullgraph=True. We should find a way to detect this case.
        if query_length > 1 and not is_tracing:
            expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
                expanded_4d_mask, attention_mask, unmasked_value=0.0
            )

    return expanded_4d_mask
