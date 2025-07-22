"""Model management for Lobster MCP server."""

import logging
from typing import Any

import torch

try:
    from lobster.model import LobsterCBMPMLM, LobsterPMLM
except ImportError as e:
    raise ImportError("Please install lobster: pip install -e .") from e

from .config import AVAILABLE_MODELS

logger = logging.getLogger("lobster-fastmcp-server")


class ModelManager:
    """Manages loading and caching of Lobster models."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models: dict[str, LobsterPMLM | LobsterCBMPMLM] = {}
        logger.info(f"Initialized ModelManager on device: {self.device}")

    def get_or_load_model(self, model_name: str, model_type: str) -> LobsterPMLM | LobsterCBMPMLM:
        """Load model if not already cached."""
        cache_key = f"{model_type}_{model_name}"

        if cache_key not in self.loaded_models:
            logger.info(f"Loading model {model_name} of type {model_type}")

            if model_type == "masked_lm":
                if model_name not in AVAILABLE_MODELS["masked_lm"]:
                    raise ValueError(f"Unknown masked LM model: {model_name}")
                model_path = AVAILABLE_MODELS["masked_lm"][model_name]
                model = LobsterPMLM(model_path).to(self.device)

            elif model_type == "concept_bottleneck":
                if model_name not in AVAILABLE_MODELS["concept_bottleneck"]:
                    raise ValueError(f"Unknown concept bottleneck model: {model_name}")
                model_path = AVAILABLE_MODELS["concept_bottleneck"][model_name]
                model = LobsterCBMPMLM(model_path).to(self.device)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.eval()
            self.loaded_models[cache_key] = model
            logger.info(f"Successfully loaded {cache_key}")

        return self.loaded_models[cache_key]

    def get_device_info(self) -> dict[str, Any]:
        """Get information about the current device."""
        return {"device": self.device}
