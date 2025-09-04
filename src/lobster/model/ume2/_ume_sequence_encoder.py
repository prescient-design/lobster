from dataclasses import dataclass
import logging
from typing import Literal
import torch.nn as nn
from torch import Tensor
from ..neobert import NeoBERTModule

logger = logging.getLogger(__name__)

# class UMESequenceStructureEncoder(nn.Module):
#     # Sequence and structure encoder that uses structural tokens
#     pass

# class UMESequenceAwareEncoder(nn.Module):
#     # Sequence-only encoder that has been aligned with structure encoder
#     pass


@dataclass
class AuxiliaryTask:
    name: str
    task_type: Literal["regression", "classification"]
    pooling: Literal["cls", "mean"] = "mean"
    output_dim: int
    hidden_size: int | None = None
    dropout: float = 0.1
    num_layers: int = 2
    loss_weight: float = 1.0

    def __post_init__(self):
        if self.pooling not in ["cls", "mean"]:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        if self.loss_weight <= 0 or self.loss_weight > 1:
            raise ValueError(f"Loss weight must be between 0 and 1: {self.loss_weight}")

        if self.task_type not in ["regression"]:
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
    ):
        super().__init__()
        self.task_name = task_name
        self.hidden_size = hidden_size if hidden_size is not None else input_dim // 2
        self.dropout = dropout
        self.num_layers = num_layers

        layers = []
        current_dim = input_dim

        for i in range(self.num_layers):
            layers.extend([nn.Linear(current_dim, self.hidden_size), nn.ReLU(), nn.Dropout(self.dropout)])
            current_dim = self.hidden_size

        layers.append(nn.Linear(self.hidden_size, output_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class UMESequenceEncoderModule(nn.Module):
    def __init__(self, auxiliary_tasks: list[AuxiliaryTask] | None = None, **kwargs) -> None:
        super().__init__()

        self.model = NeoBERTModule(**kwargs)

        if auxiliary_tasks is not None:
            if not all(task.task_type == "regression" for task in auxiliary_tasks):
                raise NotImplementedError("Only regression tasks are currently supported for auxiliary tasks in UME-2")

            print(f"Creating auxiliary tasks with model hidden_size: {self.model.config.hidden_size}")
            self.auxiliary_tasks = nn.ModuleDict(
                {
                    task.name: AuxiliaryRegressionTaskHead(
                        input_dim=self.model.config.hidden_size,
                        output_dim=task.output_dim,
                        task_name=task.name,
                        hidden_size=task.hidden_size,
                        dropout=task.dropout,
                        num_layers=task.num_layers,
                    )
                    for task in auxiliary_tasks
                }
            )

        else:
            self.auxiliary_tasks = None

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, return_auxiliary_tasks: bool = False, **kwargs
    ) -> Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if self.auxiliary_tasks is not None and return_auxiliary_tasks:
            for task_name, task_head in self.auxiliary_tasks.items():
                embeddings = output["last_hidden_state"]

                match task_head.pooling:
                    case "cls":
                        embeddings = embeddings[:, 0, :]
                    case "mean":
                        embeddings = embeddings.mean(dim=1)

                output[task_name] = task_head(embeddings)

        return output
