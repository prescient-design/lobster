"""Intervention tools for Lobster MCP server."""

import logging
from typing import Any

from ..models import ModelManager
from ..schemas import InterventionRequest

logger = logging.getLogger("lobster-fastmcp-server")


def intervene_on_sequence(request: InterventionRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Perform concept intervention on a sequence."""
    try:
        model = model_manager.get_or_load_model(request.model_name, "concept_bottleneck")

        # Perform intervention
        results = model.intervene_on_sequences(
            [request.sequence], request.concept, edits=request.edits, intervention_type=request.intervention_type
        )

        new_sequence = results[0] if results else None

        # Calculate edit distance if possible
        edit_distance = None
        try:
            import Levenshtein

            edit_distance = Levenshtein.distance(request.sequence, new_sequence)
        except ImportError:
            logger.warning("Levenshtein not available for edit distance calculation")

        return {
            "original_sequence": request.sequence,
            "modified_sequence": new_sequence,
            "concept": request.concept,
            "intervention_type": request.intervention_type,
            "num_edits": request.edits,
            "edit_distance": edit_distance,
            "model_used": f"concept_bottleneck_{request.model_name}",
        }

    except Exception as e:
        logger.error(f"Error performing intervention: {str(e)}")
        raise
