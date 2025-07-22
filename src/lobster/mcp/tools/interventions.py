"""Intervention tools for Lobster MCP server."""

import logging
from typing import Any

from ..models import ModelManager
from ..schemas import InterventionRequest

logger = logging.getLogger("lobster-fastmcp-server")


def intervene_on_sequence(request: InterventionRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Perform concept intervention on a sequence.

    This function performs concept-based interventions on biological sequences using
    a concept bottleneck model. Concept interventions allow for targeted modifications
    of sequences to increase or decrease the presence of specific interpretable
    concepts, providing insights into how these concepts influence sequence properties.

    Parameters
    ----------
    request : InterventionRequest
        The request object containing all parameters for the intervention.
        Must have the following attributes:
        - model_name (str): Name of the concept bottleneck model to use
        - sequence (str): The biological sequence to intervene on
        - concept (str): The specific concept to target for intervention
        - edits (int): Number of edits to perform during intervention
        - intervention_type (str): Type of intervention ('increase' or 'decrease')
    model_manager : ModelManager
        The model manager instance responsible for loading and managing models.
        Used to retrieve the specified concept bottleneck model.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the intervention results with the following keys:
        - original_sequence (str): The input sequence before intervention
        - modified_sequence (str or None): The sequence after intervention
        - concept (str): The concept that was targeted
        - intervention_type (str): The type of intervention performed
        - num_edits (int): Number of edits that were applied
        - edit_distance (int or None): Levenshtein distance between original and modified sequences
        - model_used (str): Identifier of the model used for intervention

    Raises
    ------
    Exception
        If there is an error loading the model or performing the intervention.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> request = InterventionRequest(
    ...     model_name="protein_concept_model",
    ...     sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...     concept="hydrophobicity",
    ...     edits=3,
    ...     intervention_type="increase"
    ... )
    >>>
    >>> # The function would be called by the MCP server
    >>> result = intervene_on_sequence(request, model_manager)
    >>>
    >>> # Example response structure
    >>> {
    ...     "original_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...     "modified_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...     "concept": "hydrophobicity",
    ...     "intervention_type": "increase",
    ...     "num_edits": 3,
    ...     "edit_distance": 2,
    ...     "model_used": "concept_bottleneck_protein_concept_model"
    ... }

    Notes
    -----
    - Concept interventions modify sequences to increase or decrease the presence
      of specific interpretable concepts learned by the model
    - The intervention_type should be either 'increase' or 'decrease' to specify
      the direction of the concept modification
    - Edit distance calculation requires the Levenshtein library; if not available,
      edit_distance will be None but the intervention will still be performed
    - The number of edits specifies how many modifications the model should attempt
      to make during the intervention process
    - This tool is useful for understanding how specific concepts influence
      sequence properties and for generating concept-guided sequence variants
    """
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
