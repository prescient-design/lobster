"""Concept-related tools for Lobster MCP server."""

import logging
from typing import Any

import torch

from ..models import ModelManager
from ..schemas import SequenceConceptsRequest, SupportedConceptsRequest

logger = logging.getLogger("lobster-fastmcp-server")


def get_sequence_concepts(request: SequenceConceptsRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Get concept predictions from a concept bottleneck model.

    This function takes a sequence of biological sequences and returns their
    corresponding concept predictions and embeddings using a concept bottleneck model.
    The concept bottleneck model learns interpretable concepts that can be used
    to understand the model's decision-making process.

    Parameters
    ----------
    request : SequenceConceptsRequest
        The request object containing the model name and sequences to analyze.
        Must have the following attributes:
        - model_name (str): Name of the concept bottleneck model to use
        - sequences (List[str]): List of biological sequences to analyze
    model_manager : ModelManager
        The model manager instance responsible for loading and managing models.
        Used to retrieve the specified concept bottleneck model.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the concept analysis results with the following keys:
        - concepts (List[List[float]]): Concept predictions for each sequence
        - concept_embeddings (List[List[float]]): Concept embeddings for each sequence
        - num_sequences (int): Number of sequences analyzed
        - num_concepts (int): Number of concepts in the model
        - model_used (str): Identifier of the model used for analysis

    Raises
    ------
    Exception
        If there is an error loading the model or processing the sequences.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> request = SequenceConceptsRequest(
    ...     model_name="protein_concept_model",
    ...     sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    ... )
    >>>
    >>> # The function would be called by the MCP server
    >>> result = get_sequence_concepts(request, model_manager)
    >>>
    >>> # Example response structure
    >>> {
    ...     "concepts": [[0.8, 0.2, 0.9, 0.1, 0.7]],
    ...     "concept_embeddings": [[0.5, 0.3, 0.8, 0.2, 0.6]],
    ...     "num_sequences": 1,
    ...     "num_concepts": 5,
    ...     "model_used": "concept_bottleneck_protein_concept_model"
    ... }

    Notes
    -----
    - The function uses torch.no_grad() for inference to save memory
    - Concept predictions are returned as probabilities or scores
    - Concept embeddings represent the learned concept representations
    - All tensors are converted to CPU and then to Python lists for JSON serialization
    """
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
    """Get list of supported concepts for a concept bottleneck model.

    This function retrieves the list of concepts that a specific concept bottleneck
    model has been trained to recognize. This is useful for understanding what
    interpretable features the model can detect and for validating concept requests.

    Parameters
    ----------
    request : SupportedConceptsRequest
        The request object containing the model name to query.
        Must have the following attributes:
        - model_name (str): Name of the concept bottleneck model to query
    model_manager : ModelManager
        The model manager instance responsible for loading and managing models.
        Used to retrieve the specified concept bottleneck model.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the supported concepts information with the following keys:
        - supported_concepts (List[str] or Any): List of concept names/identifiers
        - num_concepts (int or None): Number of supported concepts (None if concepts is not a list)
        - model_used (str): Identifier of the model used for querying

    Raises
    ------
    Exception
        If there is an error loading the model or retrieving the concept list.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> request = SupportedConceptsRequest(
    ...     model_name="protein_concept_model"
    ... )
    >>>
    >>> # The function would be called by the MCP server
    >>> result = get_supported_concepts(request, model_manager)
    >>>
    >>> # Example response structure
    >>> {
    ...     "supported_concepts": [
    ...         "hydrophobicity",
    ...         "secondary_structure",
    ...         "binding_site",
    ...         "active_site",
    ...         "transmembrane_region"
    ...     ],
    ...     "num_concepts": 5,
    ...     "model_used": "concept_bottleneck_protein_concept_model"
    ... }

    Notes
    -----
    - This function is useful for discovering available concepts before making
      concept predictions on sequences
    - The concept names returned can be used to interpret the results from
      get_sequence_concepts()
    - The function handles cases where the concept list might not be a standard list
      by setting num_concepts to None in such cases
    """
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
