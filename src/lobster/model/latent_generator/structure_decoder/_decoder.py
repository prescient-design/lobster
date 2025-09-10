from torch import nn
import abc


# base class for decoders
class BaseDecoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def preprocess(self, x):
        """Implement preprocessing proteinBBT to proper decoder format."""
        ...

    @abc.abstractmethod
    def forward(self, x_noise, x_quant, t, mask):
        """Implement decoding input.
        should output coordinates"""
        ...

    @abc.abstractmethod
    def get_output_dim(self):
        """Return the output dimension of the decoder."""
        ...
