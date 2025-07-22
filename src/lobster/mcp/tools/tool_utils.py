"""Utility tools for Lobster MCP server."""

import logging
from typing import Any

import torch

from ..models import AVAILABLE_MODELS, ModelManager
from ..schemas import NaturalnessRequest

logger = logging.getLogger("lobster-fastmcp-server")


def list_available_models(model_manager: ModelManager) -> dict[str, Any]:
    """List all available pretrained Lobster models and current device information.

    This function provides information about all available pretrained models in the
    Lobster framework, along with current device configuration. This is useful for
    discovering which models can be used and understanding the computational setup.

    Parameters
    ----------
    model_manager : ModelManager
        The model manager instance that contains device information and access
        to the available models registry.

    Returns
    -------
    dict[str, Any]
        A dictionary containing model and device information with the following keys:
        - available_models (dict): Dictionary mapping model names to their configurations
          and metadata (model type, description, etc.)
        - device (str): Current device being used (e.g., 'cuda:0', 'cpu')
        - device_type (str): Type of device ('cuda', 'cpu', 'mps')
        - Additional device-specific information from model_manager.get_device_info()

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # The function would be called by the MCP server
    >>> result = list_available_models(model_manager)
    >>>
    >>> # Example response structure
    >>> {
    ...     "available_models": {
    ...         "protein_transformer": {
    ...             "type": "transformer",
    ...             "description": "Protein sequence transformer model",
    ...             "version": "1.0.0"
    ...         },
    ...         "dna_cnn": {
    ...             "type": "cnn",
    ...             "description": "DNA sequence CNN model",
    ...             "version": "1.0.0"
    ...         }
    ...     },
    ...     "device": "cuda:0",
    ...     "device_type": "cuda",
    ...     "gpu_memory_available": "8GB"
    ... }

    Notes
    -----
    - This function is useful for discovering available models before making
      specific model requests
    - Device information helps understand computational capabilities and constraints
    - The available_models dictionary can be used to validate model names in other requests
    - This tool is typically called first to understand the system capabilities
    """
    return {"available_models": AVAILABLE_MODELS, **model_manager.get_device_info()}


def compute_naturalness(request: NaturalnessRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Compute naturalness/likelihood scores for biological sequences.

    This function calculates how "natural" or likely biological sequences are according
    to a trained model. Naturalness scores indicate how well sequences conform to the
    patterns learned by the model, which can be useful for sequence validation,
    quality assessment, or identifying potentially problematic sequences.

    Parameters
    ----------
    request : NaturalnessRequest
        The request object containing parameters for naturalness computation.
        Must have the following attributes:
        - model_name (str): Name of the model to use for naturalness computation
        - model_type (str): Type of model (e.g., 'transformer', 'cnn', 'lstm')
        - sequences (List[str]): List of biological sequences to evaluate
    model_manager : ModelManager
        The model manager instance responsible for loading and managing models.
        Used to retrieve the specified model for naturalness computation.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the naturalness computation results with the following keys:
        - sequences (List[str]): The input sequences that were evaluated
        - scores (List[float]): Naturalness/likelihood scores for each sequence
          (higher scores indicate more natural/likely sequences)
        - model_used (str): Identifier of the model used for computation

    Raises
    ------
    ValueError
        If the specified model does not support naturalness or likelihood computation
        (i.e., lacks 'naturalness' or 'likelihood' methods).
    Exception
        If there is an error loading the model or computing the scores.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> request = NaturalnessRequest(
    ...     model_name="protein_transformer",
    ...     model_type="transformer",
    ...     sequences=[
    ...         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    ...     ]
    ... )
    >>>
    >>> # The function would be called by the MCP server
    >>> result = compute_naturalness(request, model_manager)
    >>>
    >>> # Example response structure
    >>> {
    ...     "sequences": [
    ...         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    ...     ],
    ...     "scores": [0.85, 0.12],
    ...     "model_used": "transformer_protein_transformer"
    ... }

    Notes
    -----
    - The function first tries to use the model's 'naturalness' method, then falls back
      to 'likelihood' if naturalness is not available
    - Scores are typically between 0 and 1, where higher scores indicate more natural
      sequences according to the model
    - All scores are converted to Python lists for JSON serialization
    - This tool is useful for:
      * Validating generated sequences
      * Quality control of biological sequences
      * Identifying potentially problematic or artificial sequences
      * Understanding model confidence in sequence predictions
    """
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
