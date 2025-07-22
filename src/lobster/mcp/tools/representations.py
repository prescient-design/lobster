"""Sequence representation tools for Lobster MCP server."""

import logging
from typing import Any

import torch

from ..models import ModelManager
from ..schemas import SequenceRepresentationRequest

logger = logging.getLogger("lobster-fastmcp-server")


def get_sequence_representations(request: SequenceRepresentationRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Get sequence representations from a model."""
    try:
        model = model_manager.get_or_load_model(request.model_name, request.model_type)

        with torch.no_grad():
            # Get latent representations
            representations = model.sequences_to_latents(request.sequences)[-1]  # Last layer

            if request.representation_type == "cls":
                # CLS token representation
                embeddings = representations[:, 0, :].cpu().numpy()
            elif request.representation_type == "pooled":
                # Mean pooled representation
                embeddings = torch.mean(representations, dim=1).cpu().numpy()
            elif request.representation_type == "full":
                # Full sequence representations
                embeddings = representations.cpu().numpy()
            else:
                raise ValueError(f"Unknown representation type: {request.representation_type}")

            return {
                "embeddings": embeddings.tolist(),
                "embedding_dimension": embeddings.shape[-1],
                "num_sequences": len(request.sequences),
                "representation_type": request.representation_type,
                "model_used": f"{request.model_type}_{request.model_name}",
            }

    except Exception as e:
        logger.error(f"Error getting representations: {str(e)}")
        raise
