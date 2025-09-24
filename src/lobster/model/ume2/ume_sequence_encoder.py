import logging
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from lobster.constants import Modality, ModalityType
from lobster.model.utils import _detect_modality
from lobster.tokenization import get_ume_tokenizer_transforms

from ..neobert import NeoBERTModule
from ._checkpoint_utils import load_checkpoint_from_s3_uri_or_local_path, map_checkpoint_keys
from .auxiliary_tasks import AuxiliaryRegressionTaskHead, AuxiliaryTask

logger = logging.getLogger(__name__)


class UMESequenceEncoderModule(nn.Module):
    def __init__(
        self,
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        use_shared_tokenizer: bool = False,
        cache_dir: str | None = None,
        **neobert_kwargs,
    ) -> None:
        super().__init__()

        self.neobert = NeoBERTModule(**neobert_kwargs)
        self.use_shared_tokenizer = use_shared_tokenizer
        self.cache_dir = cache_dir

        if auxiliary_tasks is None:
            self.auxiliary_tasks = None
        else:
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

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, *, device: str | None = None, cache_dir: str | None = None, **kwargs
    ) -> "UMESequenceEncoderModule":
        """Utility function to load state_dict and hyper_parameters from UMESequenceEncoderLightningModule checkpoint."""

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
        self, input_ids: Tensor, attention_mask: Tensor, return_auxiliary_tasks: bool = False, **kwargs
    ) -> Tensor:
        output = self.neobert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if self.auxiliary_tasks is not None and return_auxiliary_tasks:
            for task_name, task_head in self.auxiliary_tasks.items():
                embeddings = output["last_hidden_state"]
                output[task_name] = task_head(embeddings)

        return output

    def embed(self, inputs: dict[str, Tensor], aggregate: bool = True, ignore_padding: bool = True, **kwargs) -> Tensor:
        return self.neobert.embed(inputs=inputs, aggregate=aggregate, ignore_padding=ignore_padding, **kwargs)

    def embed_sequences(
        self, sequences: Sequence[str] | str, modality: ModalityType | Modality = None, aggregate: bool = True
    ) -> Tensor:
        if isinstance(sequences, str):
            sequences = [sequences]

        if modality is None:
            modality = set([_detect_modality(sequence) for sequence in sequences])

            if len(modality) > 1:
                raise NotImplementedError(
                    f"Multiple modalities ({modality}) detected which is not currently supported for UME-2"
                )

            modality = modality.pop()

        modality = Modality(modality) if isinstance(modality, str) else modality

        tokenizer_transform = get_ume_tokenizer_transforms(
            use_shared_tokenizer=self.use_shared_tokenizer, max_length=self.neobert.config.max_length
        )[modality]
        encoded_batch = tokenizer_transform(sequences)

        return self.embed(encoded_batch, aggregate=aggregate)


def get_device() -> str:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
