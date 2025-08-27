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
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        max_length: int = 1024,
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
