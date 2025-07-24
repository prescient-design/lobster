"""Intervention tools for Lobster MCP server using FastMCP best practices."""

import logging
from typing import Literal

from .tool_utils import _load_model

logger = logging.getLogger("lobster-fastmcp-server")


def intervene_on_sequence(
    model_name: str,
    sequence: str,
    concept: str,
    edits: int = 5,
    intervention_type: Literal["positive", "negative"] = "negative",
) -> dict:
    """Perform concept intervention on a sequence.

    This function performs concept-based interventions on biological sequences using
    a concept bottleneck model. Concept interventions allow for targeted modifications
    of sequences to increase or decrease the presence of specific interpretable
    concepts, providing insights into how these concepts influence sequence properties.

    Parameters
    ----------
    model_name : str
        Name of the concept bottleneck model to use
    sequence : str
        The biological sequence to intervene on
    concept : str
        The specific concept to target for intervention
    edits : int, default=5
        Number of edits to perform during intervention
    intervention_type : str, default="negative"
        Type of intervention ('positive' or 'negative')

    Returns
    -------
    dict
        A dictionary containing the intervention results with the following fields:
        - original_sequence: The input sequence before intervention
        - modified_sequence: The sequence after intervention (None if intervention failed)
        - concept: The concept that was targeted
        - intervention_type: The type of intervention performed
        - num_edits: Number of edits that were applied
        - edit_distance: Levenshtein distance between original and modified sequences (None if Levenshtein not available)
        - model_used: Identifier of the model used for intervention

    Raises
    ------
    Exception
        If there is an error loading the model or performing the intervention.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> result = intervene_on_sequence(
    ...     model_name="cb_lobster_24M",
    ...     sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...     concept="hydrophobicity",
    ...     edits=3,
    ...     intervention_type="positive"
    ... )
    >>>
    >>> # Example response structure
    >>> {
    ...     "original_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...     "modified_sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...     "concept": "hydrophobicity",
    ...     "intervention_type": "positive",
    ...     "num_edits": 3,
    ...     "edit_distance": 2,
    ...     "model_used": "concept_bottleneck_cb_lobster_24M"
    ... }

    Notes
    -----
    - Concept interventions modify sequences to increase or decrease the presence
      of specific interpretable concepts learned by the model
    - The intervention_type should be either 'positive' or 'negative' to specify
      the direction of the concept modification
    - Edit distance calculation requires the Levenshtein library; if not available,
      edit_distance will be None but the intervention will still be performed
    - The number of edits specifies how many modifications the model should attempt
      to make during the intervention process
    - This tool is useful for understanding how specific concepts influence
      sequence properties and for generating concept-guided sequence variants
    """
    try:
        model = _load_model(model_name, "concept_bottleneck")

        # Perform intervention
        results = model.intervene_on_sequences([sequence], concept, edits=edits, intervention_type=intervention_type)

        new_sequence = results[0] if results else None

        # Calculate edit distance if possible
        edit_distance = None
        try:
            import Levenshtein

            edit_distance = Levenshtein.distance(sequence, new_sequence)
        except ImportError:
            logger.warning("Levenshtein not available for edit distance calculation")

        return {
            "original_sequence": sequence,
            "modified_sequence": new_sequence,
            "concept": concept,
            "intervention_type": intervention_type,
            "num_edits": edits,
            "edit_distance": edit_distance,
            "model_used": f"concept_bottleneck_{model_name}",
        }

    except Exception as e:
        logger.error(f"Error performing intervention: {str(e)}")
        raise
