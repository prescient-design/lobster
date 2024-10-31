import torch
import torch.nn.functional as F
from torch import nn

""" Helpers for Pooling """


class MaskedConv1d(nn.Conv1d):
    """
    A masked 1-dimensional convolution layer.
    # Taken from MSR: https://github.com/microsoft/protein-sequence-models/blob/0eecb6e22dd20229e5fe718878c9aab379fbb795/sequence_models/structure.py#L53

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1d(nn.Module):
    # Taken from MSR: https://github.com/microsoft/protein-sequence-models/blob/0eecb6e22dd20229e5fe718878c9aab379fbb795/sequence_models/structure.py#L53
    def __init__(self, in_dim):
        super().__init__()
        self.layer = MaskedConv1d(in_dim, 1, 1)  # Get one output channel, kernel size 1

    def forward(self, x, input_mask=None):
        n, ell, _ = x.shape  # [batch x sequence(751) x embedding (1280)]
        attn = self.layer(x)  # [batch x sequence x 1] --> 1D conv squashes hidden dim
        attn = attn.view(n, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(), float("-inf"))  # fill masked w/ -infs
        attn = F.softmax(attn, dim=-1).view(n, -1, 1)  # take softmax across seq len dim
        out = (attn * x).sum(dim=1)
        return out


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor = None) -> torch.Tensor:
        pass
