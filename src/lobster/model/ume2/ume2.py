from collections.abc import Sequence
from torch import Tensor, nn
import torch
from torch.nn import ModuleDict
from lobster.constants import Modality, to_modality
from lobster.model.ume2 import UMESequenceEncoderModule

import logging


logger = logging.getLogger(__name__)


class UME2(nn.Module):
    SUPPORTED_MODALITIES = [Modality.AMINO_ACID, Modality.SMILES]

    def __init__(
        self,
        encoder_kwargs: dict[str | Modality, dict] | None = None,
        ckpt_path: str | None = None,
    ):
        super().__init__()

        encoder_kwargs = encoder_kwargs or {}
        self.encoder_kwargs = {to_modality(key): value for key, value in encoder_kwargs.items()}

        self.molecular_encoders = ModuleDict({})

        for modality in self.SUPPORTED_MODALITIES:
            kwargs = self.encoder_kwargs.get(modality, {})
            encoder = UMESequenceEncoderModule(**kwargs)
            self.molecular_encoders[modality] = encoder

        # hidden_dims = self._validate_hidden_dims()
        # self.hidden_dim = hidden_dims[0]

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoints: dict[str | Modality, str],
        cache_dir: str | None = "data2/ume/checkpoints",
    ) -> "UME2":
        """Create UME2 instance from checkpoints.

        Parameters
        ----------
        checkpoints : dict[str | Modality, str]
            Dictionary mapping modalities to checkpoint paths
            encoder_kwargs : dict[str | Modality, dict], optional
            Additional kwargs for encoder initialization
        cache_dir : str, optional
            Cache directory for checkpoint loading
        Returns
        -------
        UME2
            Initialized UME2 instance with loaded checkpoints
        """
        checkpoints = {to_modality(key): value for key, value in checkpoints.items()}

        instance = cls()

        # Load checkpoints and replace encoders
        for modality, checkpoint_path in checkpoints.items():
            encoder = UMESequenceEncoderModule.load_from_checkpoint(checkpoint_path, cache_dir=cache_dir)
            instance.molecular_encoders[modality] = encoder
            logger.info(f"Loaded encoder for modality {modality} from checkpoint {checkpoint_path}")

        hidden_dims = instance._validate_hidden_dims()
        instance.hidden_dim = hidden_dims[0]

        return instance

    def _validate_hidden_dims(self):
        hidden_dims = {name: encoder.neobert.config.hidden_size for name, encoder in self.molecular_encoders.items()}

        if len(set(hidden_dims.values())) != 1:
            raise ValueError(
                f"Expected all molecular encoders to have the same hidden dimension, but got {hidden_dims}"
            )
        return list(hidden_dims.values())

    def ensure_2d(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        if input_ids.dim() == 3 and input_ids.shape[1] == 1:
            input_ids = input_ids.squeeze(1)
        if attention_mask.dim() == 3 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask.squeeze(1)

        input_ids = input_ids.to(self.device())
        attention_mask = attention_mask.to(self.device())

        return input_ids, attention_mask

    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        modality: Modality | str,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> dict[str, Tensor | list[Tensor]]:
        """
        Encode inputs with the appropriate encoder for the given modality.

        Parameters
        ----------
        input_ids : Tensor
            Input IDs to encode.
        attention_mask : Tensor
            Attention mask to encode.
        modality : Modality | str
            Modality to encode.
        output_hidden_states : bool, default True
            Whether to return hidden states.
        output_attentions : bool, default False
            Whether to return attentions.

        Returns
        -------
        dict[str, Tensor | list[Tensor]]
            Dictionary containing the encoded inputs.
            Keys:
                `last_hidden_state` : Tensor
                    The last hidden state of the encoder.
                    Has shape (batch_size, sequence_length, hidden_size)
                `hidden_states` : list[Tensor]
                    The hidden states of the encoder with n=num_hidden_layers layers
                    for each encoder
                    Only returned if output_hidden_states is True
                    Each item has shape (batch_size, sequence_length, hidden_size)
                `attentions` : list[Tensor]
                    The attentions of the encoder with n=num_hidden_layers layers
                    for each encoder
                    Only returned if output_attentions is True
                    Each item has shape (batch_size, num_heads, sequence_length, sequence_length)
        """
        modality = to_modality(modality)

        if modality not in self.SUPPORTED_MODALITIES:
            raise ValueError(f"Unsupported modality: {modality}. Expected one of: {self.SUPPORTED_MODALITIES}")

        input_ids, attention_mask = self.ensure_2d(input_ids, attention_mask)
        batch_size, sequence_length = input_ids.shape

        encoder = self.molecular_encoders[modality]

        output = encoder.neobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        assert output["last_hidden_state"].shape == (batch_size, sequence_length, self.hidden_dim)

        return output

    def embed_sequences(
        self,
        sequences: Sequence[str] | str,
        modality: Modality | str,
        *,
        aggregate: bool = True,
    ) -> Tensor:
        if isinstance(sequences, str):
            sequences = [sequences]

        modality = to_modality(modality)

        encoder = self.molecular_encoders[modality]

        return encoder.embed_sequences(sequences, modality=modality, aggregate=aggregate)
