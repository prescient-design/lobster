from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


from ._activation import get_act_fn
from ._initialization import ModuleType, init_weights
from ._normalization import get_norm_layer

from ._layers import (
    get_encoder_layer,
)

from ._embedding import (
    get_embedding_layer,
)

from ._config import FlexBertConfig

def _count_parameters(model: nn.Module, trainable: bool = True) -> int:
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


class FlexBertModel(torch.nn.Module):
    """Overall BERT model.

    Args:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controlled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.embeddings = get_embedding_layer(config)
        self.encoder = get_encoder_layer(config)
        if config.final_norm:
            # if we use prenorm attention we need to add a final norm
            self.final_norm = get_norm_layer(config)
        else:
            self.final_norm = None
        self.unpad_embeddings = config.unpad_embeddings

    def post_init(self):
        self._init_weights(reset_params=False)
        self._backward_compatibility_gradient_checkpointing()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None, "input_ids or inputs_embeds must be provided"
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            if position_ids is None and cu_seqlens is not None:
                # Generate position ids fresh for every sequence by using cu_seqlens so that for every sequence its
                # positional embeddings start from 0 as opposed to being fully sequential across sequences (i.e., we
                # want to make sure all sequences are indeed calculated as if they were separate calls to the model).
                assert cu_seqlens.dim() == 1, "cu_seqlens must be 1D tensor"
                seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
                position_ids = torch.cat([torch.arange(seqlen, device=input_ids.device) for seqlen in seqlens])
            embeddings = self.embeddings(input_ids, position_ids)
        else:
            embeddings = inputs_embeds

        encoder_outputs = self.encoder(
            hidden_states=embeddings,
            attention_mask=attention_mask,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.final_norm is not None:
            encoder_outputs = self.final_norm(encoder_outputs)
            
        return encoder_outputs

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            self._init_module_weights(module)
        else:
            assert isinstance(reset_params, bool)
            self.embeddings._init_weights(reset_params=reset_params)
            self.encoder._init_weights(reset_params=reset_params)

            if reset_params and self.config.final_norm:
                self.final_norm.reset_parameters()

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def get_number_parameters(self, count_embeddings: bool = True, trainable: bool = True) -> int:
        """Returns the number of parameters in the model.

        Args:
            count_embeddings: count the parameters in the embeddings layer, excluding position embeddings.
            trainable: only count trainable parameters.
        """
        params = sum([_count_parameters(layer, trainable) for layer in self.encoder.layers])
        if count_embeddings:
            params += _count_parameters(self.embeddings, trainable)
            if hasattr(self.embeddings, "position_embeddings"):
                params -= _count_parameters(self.embeddings.position_embeddings, trainable)
        return params


class FlexBertPredictionHead(nn.Module):
    def __init__(self, config: FlexBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.head_pred_bias)
        self.act = get_act_fn(config.head_pred_act) if config.head_pred_act else nn.Identity()
        self.norm = (
            get_norm_layer(config, compiled_norm=config.compile_model) if config.head_pred_norm else nn.Identity()
        )

    def _init_weights(self, reset_params: bool = False):
        if reset_params:
            self.norm.reset_parameters()
        init_weights(self.config, self.dense, layer_dim=self.config.hidden_size, type_of_module=ModuleType.in_module)

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))
