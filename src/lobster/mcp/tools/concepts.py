"""Concept-related tools for Lobster MCP server."""

import logging
from typing import Any

import torch

from ..models import ModelManager
from ..schemas import SequenceConceptsRequest, SupportedConceptsRequest

logger = logging.getLogger("lobster-fastmcp-server")


def get_sequence_concepts(request: SequenceConceptsRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Get concept predictions from a concept bottleneck model."""
    try:
        model = model_manager.get_or_load_model(request.model_name, "concept_bottleneck")

        with torch.no_grad():
            concepts = model.sequences_to_concepts(request.sequences)[-1]
            concept_embeddings = model.sequences_to_concepts_emb(request.sequences)[-1]

            return {
                "concepts": concepts.cpu().numpy().tolist(),
                "concept_embeddings": concept_embeddings.cpu().numpy().tolist(),
                "num_sequences": len(request.sequences),
                "num_concepts": concepts.shape[-1],
                "model_used": f"concept_bottleneck_{request.model_name}",
            }

    except Exception as e:
        logger.error(f"Error getting concepts: {str(e)}")
        raise


def get_supported_concepts(request: SupportedConceptsRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Get list of supported concepts for a concept bottleneck model."""
    try:
        model = model_manager.get_or_load_model(request.model_name, "concept_bottleneck")
        concepts = model.list_supported_concept()

        return {
            "supported_concepts": concepts,
            "num_concepts": len(concepts) if isinstance(concepts, list) else None,
            "model_used": f"concept_bottleneck_{request.model_name}",
        }

    except Exception as e:
        logger.error(f"Error getting supported concepts: {str(e)}")
        raise
