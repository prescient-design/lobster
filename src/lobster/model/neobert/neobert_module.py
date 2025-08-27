import torch
import torch.nn as nn

from ._model import NeoBERTConfig, NeoBERT


class NeoBERTModule(nn.Module):
    def __init__(
        self,
        config: NeoBERTConfig,
        mlm_probability: float = 0.15,
        mask_replace_prob: float = 0.8,
        random_replace_prob: float = 0.1,
    ):
        super().__init__()
        self.config = config

        self.model = NeoBERT(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        max_seqlen: int = None,
        cu_seqlens: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
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

    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        output = self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.decoder(output["last_hidden_state"])

        return logits
