from transformers.configuration_utils import PretrainedConfig

SEQ2SEQ_CONFIG_ARGS = {
    "MLM_small": {
        "num_hidden_layers": 3,
        "num_attention_heads": 6,
        "intermediate_size": 64,
        "hidden_size": 72,
    },
    "MLM_med": {
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "intermediate_size": 64,
        "hidden_size": 72,
    },
    "MLM_large": {
        "num_hidden_layers": 8,
        "num_attention_heads": 12,
        "intermediate_size": 64,
        "hidden_size": 72,
    },
    "MLM_cram": {  # 731K
        "num_hidden_layers": 8,
        "num_attention_heads": 12,
        "intermediate_size": 64,
        "hidden_size": 120,
        "query_bias": False,
        "key_bias": False,
        "value_bias": False,
        "intermediate_bias": False,
    },
    "MLM_cram_med": {  # 11M
        "num_hidden_layers": 8,
        "num_attention_heads": 12,
        "intermediate_size": 1024,
        "hidden_size": 384,
        "query_bias": False,
        "key_bias": False,
        "value_bias": False,
        "intermediate_bias": False,
    },
    "MLM_cram_large": {  # 67M
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "hidden_size": 768,
        "query_bias": False,
        "key_bias": False,
        "value_bias": False,
        "intermediate_bias": False,
    },
}


class PMLMConfig(PretrainedConfig):
    r"""
    This is the configuration class for a Prescient Masked Language Model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
    ----
        vocab_size (`int`, *optional*):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.

    """

    model_type = "pmlm"

    def __init__(
        self,
        vocab_size=None,
        mask_token_id=None,
        pad_token_id=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="rotary",
        use_cache=True,
        classifier_dropout=None,
        emb_layer_norm_before=None,
        token_dropout=False,
        query_bias=True,
        key_bias=True,
        value_bias=True,
        intermediate_bias=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.key_bias = key_bias
        self.query_bias = query_bias
        self.value_bias = value_bias
        self.intermediate_bias = intermediate_bias
