"""Concept-related tools for Lobster MCP server using FastMCP best practices."""

import logging

import torch

from .tool_utils import _load_model

logger = logging.getLogger("lobster-fastmcp-server")


def get_sequence_concepts(model_name: str, sequences: list[str]) -> dict:
    """Get concept predictions from a concept bottleneck model.

    This function takes a sequence of biological sequences and returns their
    corresponding concept predictions and embeddings using a concept bottleneck model.
    The concept bottleneck model learns interpretable concepts that can be used
    to understand the model's decision-making process.

    Parameters
    ----------
    model_name : str
        Name of the concept bottleneck model to use
    sequences : List[str]
        List of biological sequences to analyze

    Returns
    -------
    dict
        A dictionary containing the concept analysis results with the following fields:
        - concepts: Concept predictions for each sequence
        - concept_embeddings: Concept embeddings for each sequence
        - num_sequences: Number of sequences analyzed
        - num_concepts: Number of concepts in the model
        - model_used: Identifier of the model used for analysis

    Raises
    ------
    Exception
        If there is an error loading the model or processing the sequences.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> result = get_sequence_concepts(
    ...     model_name="cb_lobster_24M",
    ...     sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    ... )
    >>>
    >>> # Example response structure
    >>> {
    ...     "concepts": [[0.8, 0.2, 0.9, 0.1, 0.7]],
    ...     "concept_embeddings": [[0.5, 0.3, 0.8, 0.2, 0.6]],
    ...     "num_sequences": 1,
    ...     "num_concepts": 5,
    ...     "model_used": "concept_bottleneck_cb_lobster_24M"
    ... }

    Notes
    -----
    - The function uses torch.no_grad() for inference to save memory
    - Concept predictions are returned as probabilities or scores
    - Concept embeddings represent the learned concept representations
    - All tensors are converted to CPU and then to Python lists for JSON serialization
    """
    try:
        model = _load_model(model_name, "concept_bottleneck")

        with torch.no_grad():
            concepts = model.sequences_to_concepts(sequences)[-1]
            concept_embeddings = model.sequences_to_concepts_emb(sequences)[-1]

            return {
                "concepts": concepts.cpu().numpy().tolist(),
                "concept_embeddings": concept_embeddings.cpu().numpy().tolist(),
                "num_sequences": len(sequences),
                "num_concepts": concepts.shape[-1],
                "model_used": f"concept_bottleneck_{model_name}",
            }

    except Exception as e:
        logger.error(f"Error getting concepts: {str(e)}")
        raise


def get_supported_concepts(model_name: str) -> dict:
    """Get list of supported concepts for a concept bottleneck model.

    This function retrieves the list of concepts that a specific concept bottleneck
    model has been trained to recognize. This is useful for understanding what
    interpretable features the model can detect and for validating concept requests.

    Parameters
    ----------
    model_name : str
        Name of the concept bottleneck model to query

    Returns
    -------
    dict
        A dictionary containing the supported concepts information with the following fields:
        - concepts: List of concept names/identifiers
        - model_name: Name of the model
        - num_concepts: Number of supported concepts

    Raises
    ------
    Exception
        If there is an error loading the model or retrieving the concept list.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> result = get_supported_concepts("cb_lobster_24M")
    >>>
    >>> # Example response structure
    >>> {
    ...     "concepts": [
    ...         "hydrophobicity",
    ...         "secondary_structure",
    ...         "binding_site",
    ...         "active_site",
    ...         "transmembrane_region"
    ...     ],
    ...     "num_concepts": 5,
    ...     "model_used": "concept_bottleneck_cb_lobster_24M"
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
        model = _load_model(model_name, "concept_bottleneck")
        concepts = model.list_supported_concept()

        # Handle different return types from model.list_supported_concept()
        if concepts is None:
            concepts_list = []
            num_concepts = 0
        elif isinstance(concepts, list):
            concepts_list = concepts
            num_concepts = len(concepts)
        else:
            # For non-list returns (e.g., string), wrap in list for backward compatibility
            concepts_list = [concepts] if concepts else []
            num_concepts = 1 if concepts else 0

        return {
            "concepts": concepts_list,
            "model_name": model_name,
            "num_concepts": num_concepts,
        }

    except Exception as e:
        logger.error(f"Error getting supported concepts: {str(e)}")
        raise
