import torch.nn as nn
from torch import Tensor
from typing import Literal

from ._model import NeoBERTConfig, NeoBERT
from ._config import NEOBERT_CONFIGS


class NeoBERTModule(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        norm_eps: float = 1e-06,
        vocab_size: int = 1280,
        pad_token_id: int | None = None,
        max_length: int = 512,
        model_size: Literal["mini", "small", "medium", "large"] | None = None,
    ):
        super().__init__()

        if model_size is not None:
            config_args = NEOBERT_CONFIGS[model_size]
            hidden_size = config_args["hidden_size"]
            num_hidden_layers = config_args["num_hidden_layers"]
            num_attention_heads = config_args["num_attention_heads"]
            intermediate_size = config_args["intermediate_size"]

        self.config = NeoBERTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            embedding_init_range=embedding_init_range,
            decoder_init_range=decoder_init_range,
            norm_eps=norm_eps,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            max_length=max_length,
        )
        self.model = NeoBERT(self.config)
        self.decoder = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: Tensor = None,
        attention_mask: Tensor = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> dict:
        output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs,
        )

        output_dict = {
            "last_hidden_state": output.last_hidden_state,
        }

        if output_hidden_states:
            output_dict["hidden_states"] = output.hidden_states
        if output_attentions:
            output_dict["attentions"] = output.attentions

        return output_dict

    def get_logits(
        self,
        input_ids: Tensor,
        position_ids: Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: Tensor = None,
        attention_mask: Tensor = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Tensor:
        input_ids, attention_mask = self.ensure_2d(input_ids, attention_mask)

        output = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return self.decoder(output.last_hidden_state)

    def ensure_2d(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        if input_ids.dim() == 3 and input_ids.shape[1] == 1:
            input_ids = input_ids.squeeze(1)
        if attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask.squeeze(1)

        assert input_ids.dim() == 2, "Input IDs must have shape: (batch_size, seq_len)"
        assert attention_mask.dim() == 2, "Attention mask must have shape: (batch_size, seq_len)"

        return input_ids, attention_mask

    def embed(self, inputs: dict[str, Tensor], aggregate: bool = True, ignore_padding: bool = True, **kwargs) -> Tensor:
        input_ids, attention_mask = self.ensure_2d(inputs["input_ids"], inputs["attention_mask"])

        device = next(iter(self.parameters())).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        output = self(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if not aggregate:
            return output["last_hidden_state"]

        if not ignore_padding:
            return output["last_hidden_state"].mean(dim=1)

        mask = attention_mask.to(dtype=output["last_hidden_state"].dtype).unsqueeze(-1)

        masked_embeddings = output["last_hidden_state"] * mask

        sum_embeddings = masked_embeddings.sum(dim=1)
        token_counts = mask.sum(dim=1)

        return sum_embeddings / token_counts
