from torch import Tensor, nn

from lobster.constants import Modality, to_modality
from lobster.model.ume2 import UMESequenceEncoderModule

from .symmetric_cross_attention import SymmetricCrossAttentionModule

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
        shared_decoder: bool = False,
        cache_dir: str | None = None,
    ):
        super().__init__()

        self.shared_decoder = shared_decoder
        self.freeze_molecular_encoders = freeze_molecular_encoders
        self.molecular_encoders = {}
        self.encoder_kwargs = encoder_kwargs or {}
        self.encoder_kwargs = {to_modality(key): value for key, value in self.encoder_kwargs.items()}
        self.supported_modalities = supported_modalities or self.SUPPORTED_MODALITIES
        self.supported_modalities = [to_modality(modality) for modality in self.supported_modalities]

        self.checkpoints = {to_modality(key): value for key, value in checkpoints.items()}

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

            if self.freeze_molecular_encoders:
                for param in encoder.parameters():
                    param.requires_grad = False

            self.molecular_encoders[modality] = encoder

        hidden_dims, vocab_sizes = self._validate_hidden_dims_and_vocab_sizes()

        self.hidden_dim = hidden_dims[0]

        self.interaction_module = SymmetricCrossAttentionModule(
            d_model=self.hidden_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_ffn,
            dropout=dropout,
        )

        if self.shared_decoder:
            self.decoder = nn.Linear(self.hidden_dim, vocab_sizes[0])
        else:
            self.decoder = nn.ModuleDict(
                {
                    modality: nn.Linear(self.hidden_dim, encoder.neobert.config.vocab_size)
                    for modality, encoder in self.molecular_encoders.items()
                }
            )

    def _validate_hidden_dims_and_vocab_sizes(self):
        hidden_dims = [encoder.neobert.config.hidden_size for encoder in self.molecular_encoders.values()]
        vocab_sizes = [encoder.neobert.config.vocab_size for encoder in self.molecular_encoders.values()]

        if len(set(hidden_dims)) != 1:
            raise ValueError(
                f"Expected all molecular encoders to have the same hidden dimension, but got {hidden_dims}"
            )

        if len(set(vocab_sizes)) != 1 and self.shared_decoder:
            raise ValueError(
                f"When shared_decoder=True, expected all molecular encoders to share the same vocabulary, but found different sizes: {vocab_sizes}"
            )

        return hidden_dims, vocab_sizes

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
    ):
        encoder1 = self.molecular_encoders[inputs1["modality"]]
        encoder2 = self.molecular_encoders[inputs2["modality"]]

        out1 = encoder1(inputs1["input_ids"], inputs1["attention_mask"])
        out2 = encoder2(inputs2["input_ids"], inputs2["attention_mask"])

        x1 = out1["last_hidden_state"]
        x2 = out2["last_hidden_state"]

        logger.info("Shapes before interaction module:")
        logger.info(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        x1, x2 = self.interaction_module(x1, x2)

        logger.info("Shapes after interaction module:")
        logger.info(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")

        return x1, x2

    def get_logits(self, inputs1: dict[str, Tensor], inputs2: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x1, x2 = self.forward(inputs1, inputs2)

        if self.shared_decoder:
            logits1 = self.decoder(x1)
            logits2 = self.decoder(x2)
        else:
            logits1 = self.decoder[inputs1["modality"]](x1)
            logits2 = self.decoder[inputs2["modality"]](x2)

        logger.info("Logits shapes:")
        logger.info(f"logits1 shape: {logits1.shape}, logits2 shape: {logits2.shape}")

        return logits1, logits2
