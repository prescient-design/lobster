import torch
from torch import nn

from ._pooling_layers import Attention1d


class LMClsPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self, hidden_states: torch.Tensor, input_mask=None
    ) -> torch.Tensor:  # NOTE - input_mask is a dummy variable for hydra
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LMMeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        if input_mask is not None:
            input_mask = input_mask.unsqueeze(-1)
            out = (hidden_states * input_mask).sum(dim=1)
            divisor = input_mask.sum(axis=1)
            out = out / divisor
        else:
            out = hidden_states.mean(dim=1)
        # pooled_output = self.dense(out)
        # pooled_output = self.activation(pooled_output)
        # return pooled_output
        return out


class LMWeightedMeanPooler(nn.Module):
    def __init__(self, config):
        """
        Weighted mean pooling.

        Mostly for autoregressive, decoder-only models; token embeddings are weighted
        by their position in the sequence so that later tokens have higher weights.

        Args:
        ----
            config (Config): Configuration object.

        """
        super().__init__()

        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        B, L, H = hidden_states.shape

        weights = torch.tensor([i / L for i in range(1, L + 1)]).view(1, L, 1)
        weights = weights.expand(B, -1, -1)
        weighted_states = weights * hidden_states
        weighted_sum = weighted_states.sum(dim=1)
        weighted_mean = weighted_sum / weights.sum(dim=1)

        return weighted_mean


class LMAttentionPool1D(nn.Module):
    """
    Get attention weights for each non-masked residue, then take a weighted SUM (not mean) of the hidden states
    In more detail:
        - Apply a 1D convolution to the hidden states to squash the hidden dim (treating each hidden dim as a "channel")
        - Take softmax across the sequence length dimension, ignore masked residues
        - Take a weighted sum of the hidden states using the attention weights
        - Apply a linear layer + activation to the weighted sum

    Based off of: https://github.com/J-SNACKKB/FLIP/blob/214d364f4fe0ad927c799471d7faa25a44d3eca8/baselines/models.py#L14

    """

    def __init__(self, config):
        super().__init__()
        self.attention1d = Attention1d(in_dim=config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        if input_mask is not None:
            input_mask = input_mask.unsqueeze(-1)
        attention_weighted_mean = self.attention1d(hidden_states, input_mask=input_mask)

        pooled_output = self.dense(attention_weighted_mean)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossAttentionPooler(nn.Module):
    """
    Learns a fixed-length sequence representation via cross-attention with the input token embeddings.
    The fixed-length embedding is initialized as a (batch, 1, hidden_size) tensor and is learned during
    model fine-tuning.

    To be precise, this doesn't pool but it does output a fix-length embedding in a similar fashion to other poolers.
    """

    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        pass
