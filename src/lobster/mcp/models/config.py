"""Model configuration and constants for Lobster MCP server."""

# Available pretrained models
AVAILABLE_MODELS = {
    "masked_lm": {
        "lobster_24M": "asalam91/lobster_24M",
        "lobster_150M": "asalam91/lobster_150M",
    },
    "concept_bottleneck": {
        "cb_lobster_24M": "asalam91/cb_lobster_24M",
        "cb_lobster_150M": "asalam91/cb_lobster_150M",
        "cb_lobster_650M": "asalam91/cb_lobster_650M",
        "cb_lobster_3B": "asalam91/cb_lobster_3B",
    },
}
