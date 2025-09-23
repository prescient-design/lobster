from torch import Tensor, nn

from lobster.constants import Modality
from lobster.model.ume2 import UMESequenceEncoderModule

from .symmetric_cross_attention import SymmetricCrossAttentionModule

import logging

logger = logging.getLogger(__name__)


class UMEInteraction(nn.Module):
    SUPPORTED_MODALITIES = [Modality.AMINO_ACID, Modality.SMILES]

    def __init__(
        self,
        checkpoints: dict[str | Modality, str] | None = None,
        freeze_molecular_encoders: bool = False,
        cache_dir: str | None = None,
        use_shared_tokenizer: bool = False,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_ffn: int = 2048,
        dropout: float = 0.1,
        shared_decoder: bool = False,
        encoder_kwargs: dict | None = None,
    ):
        super().__init__()

        self.shared_decoder = shared_decoder
        self.freeze_molecular_encoders = freeze_molecular_encoders
        self.checkpoints = checkpoints if checkpoints is not None else {}
        self.molecular_encoders = {}

        for modality in self.SUPPORTED_MODALITIES:
            encoder = UMESequenceEncoderModule(
                model_ckpt=self.checkpoints.get(modality),
                cache_dir=cache_dir,
                use_shared_tokenizer=use_shared_tokenizer,
                **encoder_kwargs or {},
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
