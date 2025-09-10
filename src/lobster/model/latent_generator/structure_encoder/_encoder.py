from torch import nn
import abc


# base class for encoders, that has a featurize method, forwad pass, and a featurize method
class BaseEncoder(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def featurize(self, x):
        """Implement featurizing proteinBBT to proper encoder format."""
        ...

    @abc.abstractmethod
    def forward(self, x):
        """Implement encoding input."""
        ...
