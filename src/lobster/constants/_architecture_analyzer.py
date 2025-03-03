from enum import Enum


class GPUType(Enum):
    V100 = "V100"
    A100 = "A100"
    H100 = "H100"

    @property
    def streaming_multiprocessors(self) -> int:
        if self == GPUType.V100:
            return 80
        elif self == GPUType.A100:
            return 108
        elif self == GPUType.H100:
            return 144
        else:
            raise ValueError(f"Unknown GPU type: {self}")


class ModelType(Enum):
    DECODER_ONLY = "decoder_only"  # GPT-style models
    ENCODER_ONLY = "encoder_only"  # BERT-style models
    ENCODER_DECODER = "encoder_decoder"  # T5-style models
