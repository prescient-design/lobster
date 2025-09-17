# From https://huggingface.co/chandar-lab/NeoBERT/blob/main/model.py


import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from ._config import NeoBERTConfig

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from transformers import (
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
)

from ._rotary import apply_rotary_emb, precompute_freqs_cis
from ._swiglu import SwiGLU


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: NeoBERTConfig):
        super().__init__()

        self.config = config

        # Attention
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)

        # Feedforward network
        multiple_of = 8
        intermediate_size = int(2 * config.intermediate_size / 3)
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.ffn = SwiGLU(config.hidden_size, intermediate_size, config.hidden_size, bias=False)

        # Layer norms
        self.attention_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        output_attentions: bool,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
    ):
        # Attention
        attn_output, attn_weights = self._att_block(
            self.attention_norm(x), attention_mask, freqs_cis, output_attentions, max_seqlen, cu_seqlens
        )

        # Residual
        x = x + attn_output

        # Feed-forward
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights

    def _att_block(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        output_attentions: bool,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = (
            self.qkv(x)
            .view(batch_size, seq_len, self.config.num_attention_heads, self.config.dim_head * 3)
            .chunk(3, axis=-1)
        )

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Attn block
        attn_weights = None

        # Flash attention if the tensors are packed
        if cu_seqlens is not None:
            attn = flash_attn_varlen_func(
                q=xq.squeeze(0),
                k=xk.squeeze(0),
                v=xv.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
            )
        # Eager attention if attention weights are needed in the output
        elif output_attentions:
            attn_weights = xq.permute(0, 2, 1, 3) @ xk.permute(0, 2, 3, 1) / (xq.size(-1) ** 0.5)
            if attention_mask is not None:
                attn_weights = attn_weights * attention_mask
            attn_weights = attn_weights.softmax(-1)
            attn = attn_weights @ xv.permute(0, 2, 1, 3)
            attn = attn.transpose(1, 2)
        # Fall back to SDPA otherwise
        else:
            attn = scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=attention_mask.bool(),
                dropout_p=0,
            ).transpose(1, 2)

        return self.wo(
            attn.reshape(batch_size, seq_len, self.config.num_attention_heads * self.config.dim_head)
        ), attn_weights


class NeoBERTPreTrainedModel(PreTrainedModel):
    config_class = NeoBERTConfig
    base_model_prefix = "model"
    _supports_cache_class = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-self.config.decoder_init_range, self.config.decoder_init_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.uniform_(-self.config.embedding_init_range, self.config.embedding_init_range)


class NeoBERT(NeoBERTPreTrainedModel):
    config_class = NeoBERTConfig

    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)

        self.config = config

        self.encoder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        # Ensures freqs_cis is moved to the same devices as the model. Non-persistent buffers are not saved in the state_dict.
        freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.max_length)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.transformer_encoder = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.transformer_encoder.append(EncoderBlock(config))

        self.layer_norm = nn.RMSNorm(config.hidden_size, config.norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        # Initialize
        hidden_states, attentions = [], []

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Expand and repeat: (Batch, Length) -> (Batch, Heads, Length, Length)
        if attention_mask is not None:
            try:
                attention_mask = (
                    attention_mask.unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, self.config.num_attention_heads, attention_mask.size(-1), 1)
                )
            except Exception as e:
                raise ValueError(
                    f"Attention mask shape is not correct {attention_mask.shape}. Must be (batch_size, seq_len)."
                ) from e

        # Checks to be done if inputs are packed sequences
        if cu_seqlens is not None:
            assert FLASH_ATTN_AVAILABLE, (
                "Flash-attention is not available. Please ''pip install flash_attn'', or provide un-packed sequences."
            )
            assert not output_attentions, "Output attentions is not supported when sequences are packed."
            assert max_seqlen is not None, "Missing max_seqlen. It must be provided when cu_seqlens are not None."
            assert (input_ids if input_ids is not None else inputs_embeds).shape[0] == 1, (
                "Cumulative sequence lengths are provided but inputs are not packed."
            )
            assert (input_ids if input_ids is not None else inputs_embeds).is_cuda, (
                "Packing uses an implementation of flash-attention and is only supported on GPU."
            )

        # RoPE
        freqs_cis = (
            self.freqs_cis[position_ids]
            if position_ids is not None
            else self.freqs_cis[: (input_ids if input_ids is not None else inputs_embeds).shape[1]].unsqueeze(0)
        )

        # Embedding
        x = self.encoder(input_ids) if input_ids is not None else inputs_embeds

        # Transformer encoder
        for layer in self.transformer_encoder:
            x, attn = layer(x, attention_mask, freqs_cis, output_attentions, max_seqlen, cu_seqlens)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn)

        # Final normalization layer
        x = self.layer_norm(x)

        # Return the output of the last hidden layer
        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=attentions if output_attentions else None,
        )
