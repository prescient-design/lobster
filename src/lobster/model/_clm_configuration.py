PCLM_CONFIG_ARGS = {
    "CLM_mini": {  # 33K
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "hidden_size": 32,
    },
    "CLM_bottleneck": {  # 2.4M
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "hidden_size": 16,
    },
    "CLM_11M": {  # 10.6M
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "intermediate_size": 1024,
        "hidden_size": 384,
    },
    "CLM_24M": {  # 23.6M
        "num_hidden_layers": 8,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "hidden_size": 384,
    },
    "CLM_68M": {  # 67.9M
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "hidden_size": 648,
    },
    "CLM_85M": {  # 85M
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "hidden_size": 768,
    },
    "CLM_113M": {  # 113.3M
        "num_hidden_layers": 16,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "hidden_size": 768,
    },
    "CLM_150M": {  # 150.6M
        "num_hidden_layers": 20,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "hidden_size": 816,
    },
}
