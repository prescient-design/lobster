import torch.nn as nn
from torch import Tensor

from ._model import NeoBERTConfig, NeoBERT
from ._masking import mask_tokens


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
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        max_length: int = 1024,
    ):
        super().__init__()

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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

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

    def get_masked_logits_and_labels(
        self, input_ids: Tensor, attention_mask: Tensor, **kwargs
    ) -> tuple[Tensor, Tensor]:
        """Get masked logits and labels for a batch of input IDs and attention masks.

        Parameters
        ----------
        input_ids : Tensor
            The input IDs to mask.
        attention_mask : Tensor
            The attention mask for the input IDs.

        Returns
        -------
        tuple[Tensor, Tensor]
            A tuple containing the masked logits and labels.
            Shape of logits: (batch_size, seq_len, vocab_size)
            Shape of labels: (batch_size, seq_len)
        """
        if (
            input_ids.dim() == 3
            and input_ids.shape[1] == 1
            and attention_mask.dim() == 3
            and attention_mask.shape[1] == 1
        ):
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

        masked_inputs = mask_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            mask_probability=self.mask_probability,
            special_token_ids=self.special_token_ids,
            generator=self.generator,
        )
        input_ids = masked_inputs["input_ids"]
        attention_mask = masked_inputs["attention_mask"]
        labels = masked_inputs["labels"]

        logits = self.get_logits(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        return logits, labels
