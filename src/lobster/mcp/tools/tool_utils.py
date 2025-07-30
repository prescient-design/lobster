"""Utility tools for Lobster MCP server."""

import logging
from functools import cache
from typing import Any, Literal

import torch

try:
    from lobster.model import LobsterCBMPMLM, LobsterPMLM
except ImportError as e:
    raise ImportError("Please install lobster: pip install -e .") from e

from ..models.config import AVAILABLE_MODELS

logger = logging.getLogger("lobster-fastmcp-server")


def _get_device() -> str:
    """Get the current device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@cache
def _load_model(model_name: str, model_type: str) -> LobsterPMLM | LobsterCBMPMLM:
    """Load a model directly using the AVAILABLE_MODELS config.

    Parameters
    ----------
    model_name : str
        Name of the model to load
    model_type : str
        Type of model: 'masked_lm' or 'concept_bottleneck'

    Returns
    -------
    LobsterPMLM | LobsterCBMPMLM
        The loaded model
    """
    device = _get_device()

    if model_type == "masked_lm":
        if model_name not in AVAILABLE_MODELS["masked_lm"]:
            raise ValueError(f"Unknown masked LM model: {model_name}")
        model_path = AVAILABLE_MODELS["masked_lm"][model_name]
        model = LobsterPMLM(model_path).to(device)

    elif model_type == "concept_bottleneck":
        if model_name not in AVAILABLE_MODELS["concept_bottleneck"]:
            raise ValueError(f"Unknown concept bottleneck model: {model_name}")
        model_path = AVAILABLE_MODELS["concept_bottleneck"][model_name]
        model = LobsterCBMPMLM(model_path).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model


def list_available_models() -> dict[str, Any]:
    """List all available pretrained Lobster models and current device information.

    This function provides information about all available pretrained models in the
    Lobster framework, along with current device configuration. This is useful for
    discovering which models can be used and understanding the computational setup.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing model and device information with the following fields:
        - available_models: Dictionary mapping model names to their configurations and metadata
        - device: Current device being used (e.g., 'cuda:0', 'cpu')
        - device_type: Type of device ('cuda', 'cpu', 'mps')

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # The function would be called by the MCP server
    >>> result = list_available_models()
    >>>
    >>> # Example response structure
    >>> {
    ...     "available_models": {
    ...         "masked_lm": {
    ...             "lobster_24M": "asalam91/lobster_24M",
    ...             "lobster_150M": "asalam91/lobster_150M"
    ...         },
    ...         "concept_bottleneck": {
    ...             "cb_lobster_24M": "asalam91/cb_lobster_24M",
    ...             "cb_lobster_150M": "asalam91/cb_lobster_150M",
    ...             "cb_lobster_650M": "asalam91/cb_lobster_650M",
    ...             "cb_lobster_3B": "asalam91/cb_lobster_3B"
    ...         }
    ...     },
    ...     "device": "cuda:0",
    ...     "device_type": "cuda"
    ... }

    Notes
    -----
    - This function is useful for discovering available models before making
      specific model requests
    - Device information helps understand computational capabilities and constraints
    - The available_models dictionary can be used to validate model names in other requests
    - This tool is typically called first to understand the system capabilities
    """
    device = _get_device()
    device_type = "cuda" if device.startswith("cuda") else device

    return {"available_models": AVAILABLE_MODELS, "device": device, "device_type": device_type}


def compute_naturalness(
    model_name: str, sequences: list[str], model_type: Literal["masked_lm", "concept_bottleneck"]
) -> dict[str, Any]:
    """Compute naturalness/likelihood scores for biological sequences.

    This function calculates how "natural" or likely biological sequences are according
    to a trained model. Naturalness scores indicate how well sequences conform to the
    patterns learned by the model, which can be useful for sequence validation,
    quality assessment, or identifying potentially problematic sequences.

    Parameters
    ----------
    model_name : str
        Name of the model to use for naturalness computation
    sequences : List[str]
        List of biological sequences to evaluate
    model_type : str
        Type of model: 'masked_lm' or 'concept_bottleneck'

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the naturalness computation results with the following fields:
        - sequences: The input sequences that were evaluated
        - scores: Naturalness/likelihood scores for each sequence (higher scores indicate more natural/likely sequences)
        - model_used: Identifier of the model used for computation

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
    >>> result = compute_naturalness(
    ...     model_name="lobster_24M",
    ...     model_type="masked_lm",
    ...     sequences=[
    ...         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    ...     ]
    ... )
    >>>
    >>> # Example response structure
    >>> {
    ...     "sequences": [
    ...         "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ...         "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    ...     ],
    ...     "scores": [0.85, 0.12],
    ...     "model_used": "masked_lm_lobster_24M"
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
        model = _load_model(model_name, model_type)

        if hasattr(model, "naturalness"):
            scores = model.naturalness(sequences)
        elif hasattr(model, "likelihood"):
            scores = model.likelihood(sequences)
        else:
            raise ValueError(f"Model {model_name} does not support naturalness/likelihood computation")

        return {
            "sequences": sequences,
            "scores": scores.tolist() if torch.is_tensor(scores) else scores,
            "model_used": f"{model_type}_{model_name}",
        }

    except Exception as e:
        logger.error(f"Error computing naturalness: {str(e)}")
        raise
