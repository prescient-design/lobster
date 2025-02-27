"""Pre-set configurations for Modern BERT models.

hidden_size: Increasing it significantly increases model capacity but also computational requirements.

num_hidden_layers: More layers increase depth and capacity, but also increase computational cost linearly.

num_attention_heads: Generally, this is set to hidden_size / 64.

intermediate_size: Often set to 4 * hidden_size.
"""

FLEXBERT_CONFIG_ARGS = {
    "UME_mini": {
        "num_hidden_layers": 6,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "hidden_size": 384,
    },
    "UME_small": {
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_size": 768,
    },
    "UME_medium": {
        "num_hidden_layers": 24,
        "num_attention_heads": 20,
        "intermediate_size": 5120,
        "hidden_size": 1280,
    },
    "UME_large": {
        "num_hidden_layers": 24,
        "num_attention_heads": 25,
        "intermediate_size": 6400,
        "hidden_size": 1600,
    },
}