from torch import Tensor, nn
from torch.nn import ModuleDict
from lobster.constants import Modality, to_modality
from lobster.model.ume2 import UMESequenceEncoderModule

from ._cross_attention import SymmetricCrossAttentionModule

import logging

logger = logging.getLogger(__name__)


class UMEInteraction(nn.Module):
    SUPPORTED_MODALITIES = [Modality.AMINO_ACID, Modality.SMILES]

    def __init__(
        self,
        checkpoints: dict[str | Modality, str] | None = None,
        supported_modalities: list[str | Modality] | None = None,
        encoder_kwargs: dict[str | Modality, dict] | None = None,
        freeze_molecular_encoders: bool = False,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_ffn: int = 2048,
        dropout: float = 0.1,
        cache_dir: str | None = None,
    ):
        super().__init__()

        self.freeze_molecular_encoders = freeze_molecular_encoders

        encoder_kwargs = encoder_kwargs or {}
        self.encoder_kwargs = {to_modality(key): value for key, value in encoder_kwargs.items()}

        supported_modalities = supported_modalities or self.SUPPORTED_MODALITIES
        self.supported_modalities = [to_modality(modality) for modality in supported_modalities]

        checkpoints = checkpoints or {}
        self.checkpoints = {to_modality(key): value for key, value in checkpoints.items()}

        self.molecular_encoders = ModuleDict({})

        for modality in self.supported_modalities:
            # Load from checkpoint
            if (checkpoint := self.checkpoints.get(modality)) is not None:
                encoder = UMESequenceEncoderModule.load_from_checkpoint(checkpoint, cache_dir=cache_dir)
                logger.info(f"Loaded encoder for modality {modality} from checkpoint {checkpoint}")

            # Or initialize a new encoder
            else:
                logger.warning(f"No checkpoint provided for modality {modality}, initializing a new encoder.")

                if (kwargs := self.encoder_kwargs.get(modality)) is None:
                    logger.critical(
                        f"No checkpoint or encoder_kwargs were provided for modality {modality}. Please ensure this is intended."
                    )

                encoder = UMESequenceEncoderModule(
                    **kwargs or {},
                )

            self.molecular_encoders[modality] = encoder

        hidden_dims = self._validate_hidden_dims()
        self.hidden_dim = hidden_dims[0]

        self.interaction_module = SymmetricCrossAttentionModule(
            hidden_size=self.hidden_dim,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=dim_ffn,
            dropout=dropout,
        )

        self.decoders = nn.ModuleDict(
            {
                modality: nn.Linear(self.hidden_dim, encoder.neobert.config.vocab_size)
                for modality, encoder in self.molecular_encoders.items()
            }
        )

        if self.freeze_molecular_encoders:
            self._freeze_encoders()

    def _freeze_encoders(self):
        for encoder in self.molecular_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = False

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

        return input_ids, attention_mask

    def forward(
        self,
        inputs1: dict[str, Tensor],
        inputs2: dict[str, Tensor],
        use_cross_attention: bool = True,
    ):
        encoder1 = self.molecular_encoders[inputs1["modality"]]
        encoder2 = self.molecular_encoders[inputs2["modality"]]

        out1 = encoder1(inputs1["input_ids"], inputs1["attention_mask"])
        out2 = encoder2(inputs2["input_ids"], inputs2["attention_mask"])

        x1 = out1["last_hidden_state"]
        x2 = out2["last_hidden_state"]

        if use_cross_attention:
            x1, x2 = self.interaction_module(
                x1, x2, x1_attention_mask=inputs1["attention_mask"], x2_attention_mask=inputs2["attention_mask"]
            )

        return x1, x2

    def get_logits(
        self, inputs1: dict[str, Tensor], inputs2: dict[str, Tensor], use_cross_attention: bool = True
    ) -> tuple[Tensor, Tensor]:
        x1, x2 = self.forward(inputs1, inputs2, use_cross_attention)

        logits1 = self.decoders[inputs1["modality"]](x1)
        logits2 = self.decoders[inputs2["modality"]](x2)

        return logits1, logits2
