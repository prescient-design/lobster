import logging
from dataclasses import dataclass
from typing import Literal

import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class AuxiliaryTask:
    name: str
    output_dim: int
    task_type: Literal["regression"] = "regression"
    pooling: Literal["cls", "mean"] = "mean"
    hidden_size: int | None = None
    dropout: float = 0.1
    num_layers: int = 2
    loss_weight: float = 1.0

    def __post_init__(self):
        if self.pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        if self.loss_weight <= 0 or self.loss_weight > 1:
            raise ValueError(f"Loss weight must be between 0 and 1: {self.loss_weight}")

        if self.task_type not in {"regression"}:
            raise ValueError(f"Unsupported task type: {self.task_type}")


class AuxiliaryRegressionTaskHead(nn.Module):
    """Head for auxiliary regression tasks"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_name: str,
        hidden_size: int | None = None,
        dropout: float = 0.1,
        num_layers: int = 2,
        pooling: Literal["cls", "mean"] = "mean",
    ):
        super().__init__()
        self.task_name = task_name
        self.hidden_size = hidden_size if hidden_size is not None else input_dim // 2
        self.dropout = dropout
        self.num_layers = num_layers
        self.pooling = pooling
        layers = []
        current_dim = input_dim

        for i in range(self.num_layers):
            layers.extend([nn.Linear(current_dim, self.hidden_size), nn.ReLU(), nn.Dropout(self.dropout)])
            current_dim = self.hidden_size

        layers.append(nn.Linear(self.hidden_size, output_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        match self.pooling:
            case "cls":
                x = x[:, 0, :]
            case "mean":
                if attention_mask is None:
                    x = x.mean(dim=1)
                else:
                    mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
                    x = x * mask
                    x = x.sum(dim=1) / mask.sum(dim=1)

        return self.head(x)
