import logging
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from lobster.data import download_from_s3

from ..neobert import NeoBERTModule

logger = logging.getLogger(__name__)


def _map_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Map checkpoint keys to match current model structure.

    The checkpoint contains keys with an extra 'model.' prefix that needs to be removed.
    For example: 'model.model.encoder.weight' -> 'model.encoder.weight'

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        The original state dict from the checkpoint

    Returns
    -------
    dict[str, torch.Tensor]
        The mapped state dict with corrected keys
    """
    mapped_state_dict = {}

    for key, value in state_dict.items():
        # Remove the extra 'model.' prefix if it exists
        if key.startswith("model.model."):
            new_key = key.replace("model.model.", "model.", 1)
            mapped_state_dict[new_key] = value
            logger.debug(f"Mapped key: {key} -> {new_key}")

        elif key.startswith("model.decoder."):
            new_key = key.replace("model.decoder.", "decoder.", 1)
            mapped_state_dict[new_key] = value
            logger.debug(f"Mapped key: {key} -> {new_key}")

        else:
            mapped_state_dict[key] = value

    return mapped_state_dict


# class UMESequenceStructureEncoder(nn.Module):
#     # Sequence and structure encoder that uses structural tokens
#     pass

# class UMESequenceAwareEncoder(nn.Module):
#     # Sequence-only encoder that has been aligned with structure encoder
#     pass


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

    def forward(self, x: Tensor) -> Tensor:
        match self.pooling:
            case "cls":
                x = x[:, 0, :]
            case "mean":
                x = x.mean(dim=1)

        return self.head(x)


class UMESequenceEncoderModule(nn.Module):
    def __init__(
        self,
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        model_ckpt: str | None = None,
        cache_dir: str | None = None,
        **neobert_kwargs,
    ) -> None:
        super().__init__()

        self.neobert = NeoBERTModule(**neobert_kwargs)

        if model_ckpt is not None:
            if model_ckpt.startswith("s3://"):
                if cache_dir is None:
                    raise ValueError("cache_dir must be provided if model_ckpt is an S3 path")

                model_name = model_ckpt.split("/")[-1]
                local_filepath = os.path.join(cache_dir, model_name)

                logger.info(f"Downloading checkpoint from {model_ckpt} to {local_filepath}")
                download_from_s3(model_ckpt, local_filepath)

                model_ckpt = local_filepath

            device = next(iter(self.neobert.parameters())).device
            checkpoint = torch.load(model_ckpt, map_location=device, weights_only=False)

            logger.info(f"Loading checkpoint from {model_ckpt}")

            # Map checkpoint keys to match current model structure
            state_dict = checkpoint["state_dict"]
            mapped_state_dict = _map_checkpoint_keys(state_dict)
            self.neobert.load_state_dict(mapped_state_dict)

        if auxiliary_tasks is not None:
            if not all(task.task_type == "regression" for task in auxiliary_tasks):
                raise NotImplementedError("Only regression tasks are currently supported for auxiliary tasks in UME-2")

            self.auxiliary_tasks = nn.ModuleDict(
                {
                    task.name: AuxiliaryRegressionTaskHead(
                        input_dim=self.neobert.config.hidden_size,
                        output_dim=task.output_dim,
                        task_name=task.name,
                        hidden_size=task.hidden_size,
                        dropout=task.dropout,
                        num_layers=task.num_layers,
                        pooling=task.pooling,
                    )
                    for task in auxiliary_tasks
                }
            )

        else:
            self.auxiliary_tasks = None

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, return_auxiliary_tasks: bool = False, **kwargs
    ) -> Tensor:
        output = self.neobert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if self.auxiliary_tasks is not None and return_auxiliary_tasks:
            for task_name, task_head in self.auxiliary_tasks.items():
                embeddings = output["last_hidden_state"]
                output[task_name] = task_head(embeddings)

        return output
