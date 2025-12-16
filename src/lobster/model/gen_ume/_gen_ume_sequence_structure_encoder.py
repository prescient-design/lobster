import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


from ..neobert import NeoBERTModule
from ..ume2 import AuxiliaryRegressionTaskHead
from ..ume2._checkpoint_utils import load_checkpoint_from_s3_uri_or_local_path, map_checkpoint_keys

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


class UMESequenceStructureEncoderModule(nn.Module):
    def __init__(
        self,
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        model_ckpt: str | None = None,
        cache_dir: str | None = None,
        sequence_token_vocab_size: int = 33,
        structure_token_vocab_size: int = 258,
        sequence_token_pad_token_id: int = 1,
        structure_token_pad_token_id: int = 257,
        conditioning_input_dim: int = 1,
        **neobert_kwargs,
    ) -> None:
        super().__init__()

        self.neobert = NeoBERTModule(**neobert_kwargs)

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

        # embedding for sequence and structure tokens
        self.sequence_embedding = nn.Embedding(
            sequence_token_vocab_size, self.neobert.config.hidden_size, padding_idx=sequence_token_pad_token_id
        )
        self.structure_embedding = nn.Embedding(
            structure_token_vocab_size, self.neobert.config.hidden_size, padding_idx=structure_token_pad_token_id
        )
        self.conditioning_embedding = nn.Linear(conditioning_input_dim, self.neobert.config.hidden_size, bias=False)
        self.combine_embedding = nn.Linear(self.neobert.config.hidden_size * 3, self.neobert.config.hidden_size)

        # output for sequence and structure tokens
        self.sequence_output = nn.Linear(self.neobert.config.hidden_size, sequence_token_vocab_size)
        self.structure_output = nn.Linear(self.neobert.config.hidden_size, structure_token_vocab_size)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, *, device: str | None = None, cache_dir: str | None = None, **kwargs
    ) -> "UMESequenceStructureEncoderModule":
        """Utility function to load state_dict and hyper_parameters from UMESequenceStructureEncoderLightningModule checkpoint."""

        device = device or get_device()

        checkpoint = load_checkpoint_from_s3_uri_or_local_path(checkpoint_path, device=device, cache_dir=cache_dir)

        # Get and update hyper_parameters
        hyper_parameters = checkpoint["hyper_parameters"] or {}
        keys = ["auxiliary_tasks", "encoder_kwargs", "pad_token_id", "use_shared_tokenizer"]
        hyper_parameters = {key: value for key, value in hyper_parameters.items() if key in keys}
        hyper_parameters.update(kwargs)

        state_dict = checkpoint["state_dict"]

        encoder_kwargs = hyper_parameters.pop("encoder_kwargs", {})

        # Initialize encoder
        encoder = cls(**hyper_parameters, **encoder_kwargs)
        encoder.to(device)

        # Load state_dict
        state_dict = map_checkpoint_keys(state_dict, original_prefix="encoder.neobert.", new_prefix="")
        encoder.neobert.load_state_dict(state_dict)

        return encoder

    def forward(
        self,
        sequence_input_ids: Tensor,
        structure_input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        conditioning_tensor: Tensor | None = None,
        return_auxiliary_tasks: bool = False,
        timesteps: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        sequence_output = self.sequence_embedding(sequence_input_ids)
        structure_output = self.structure_embedding(structure_input_ids)
        conditioning_output = self.conditioning_embedding(conditioning_tensor)
        combined_output = self.combine_embedding(
            torch.cat([sequence_output, structure_output, conditioning_output], dim=-1)
        )
        # removinf position_ids becuase not properly formulated for current neo architecture
        position_ids = None
        output = self.neobert(
            input_ids=None,
            inputs_embeds=combined_output,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        sequence_output = self.sequence_output(output["last_hidden_state"])
        structure_output = self.structure_output(output["last_hidden_state"])
        output["sequence_logits"] = sequence_output
        output["structure_logits"] = structure_output

        if self.auxiliary_tasks is not None and return_auxiliary_tasks:
            for task_name, task_head in self.auxiliary_tasks.items():
                embeddings = output["last_hidden_state"]
                output[task_name] = task_head(embeddings)

        return output


def get_device() -> str:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
