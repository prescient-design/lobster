from torch import Tensor
from torch.nn import functional as F
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
