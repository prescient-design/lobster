"""Utility tools for Lobster MCP server."""

import logging
from typing import Any

import torch

from ..models import AVAILABLE_MODELS, ModelManager
from ..schemas import NaturalnessRequest

logger = logging.getLogger("lobster-fastmcp-server")


def list_available_models(model_manager: ModelManager) -> dict[str, Any]:
    """List all available pretrained Lobster models and current device."""
    return {"available_models": AVAILABLE_MODELS, **model_manager.get_device_info()}


def compute_naturalness(request: NaturalnessRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Compute naturalness/likelihood scores for sequences."""
    try:
        model = model_manager.get_or_load_model(request.model_name, request.model_type)

        if hasattr(model, "naturalness"):
            scores = model.naturalness(request.sequences)
        elif hasattr(model, "likelihood"):
            scores = model.likelihood(request.sequences)
        else:
            raise ValueError(f"Model {request.model_name} does not support naturalness/likelihood computation")

        return {
            "sequences": request.sequences,
            "scores": scores.tolist() if torch.is_tensor(scores) else scores,
            "model_used": f"{request.model_type}_{request.model_name}",
        }

    except Exception as e:
        logger.error(f"Error computing naturalness: {str(e)}")
        raise
