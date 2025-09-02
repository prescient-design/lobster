# From https://huggingface.co/chandar-lab/NeoBERT/blob/main/model.py

from transformers import PretrainedConfig


class NeoBERTConfig(PretrainedConfig):
    model_type = "neobert"

    # All config parameters must have a default value.
    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        norm_eps: float = 1e-06,
        vocab_size: int = 1280,
        pad_token_id: int = 0,
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")
        self.dim_head = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.kwargs = kwargs


# NeoBERT configs sized to match UME model parameter counts
NEOBERT_CONFIGS = {
    # Target: ~12M parameters (to match ume-mini-base-12M)
    "mini": {
        "num_hidden_layers": 14,  # Increased to get closer to 12M target
        "num_attention_heads": 4,  # 256/4 = 64 (head_dim divisible by 64)
        "intermediate_size": 1024,  # 4 * 256, divisible by 64
        "hidden_size": 256,  # Divisible by 64
    },
    # Target: ~90M parameters (to match ume-small-base-90M)
    "small": {
        "num_hidden_layers": 28,  # Increased to get closer to 90M target
        "num_attention_heads": 8,  # 512/8 = 64 (head_dim = 64)
        "intermediate_size": 2048,  # 4 * 512, divisible by 64
        "hidden_size": 512,  # Divisible by 64
    },
    # Target: ~480M parameters (to match ume-medium-base-480M)
    "medium": {
        "num_hidden_layers": 66,
        "num_attention_heads": 12,  # 768/12 = 64 (head_dim = 64)
        "intermediate_size": 3072,  # 4 * 768, divisible by 64
        "hidden_size": 768,  # Divisible by 64
    },
    # Target: ~740M parameters (to match ume-large-base-740M)
    "large": {
        "num_hidden_layers": 58,
        "num_attention_heads": 16,  # 1024/16 = 64 (head_dim = 64)
        "intermediate_size": 4096,  # 4 * 1024, divisible by 64
        "hidden_size": 1024,  # Divisible by 64
    },
}
