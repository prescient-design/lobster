"""Sequence representation tools for Lobster MCP server."""

import logging
from typing import Any

import torch

from ..models import ModelManager
from ..schemas import SequenceRepresentationRequest

logger = logging.getLogger("lobster-fastmcp-server")


def get_sequence_representations(request: SequenceRepresentationRequest, model_manager: ModelManager) -> dict[str, Any]:
    """Get sequence representations from a model.

    This function extracts learned representations (embeddings) from biological sequences
    using various types of models. The representations can be used for downstream tasks
    such as sequence similarity analysis, clustering, or as features for machine learning
    models. Different representation types provide different levels of sequence information.

    Parameters
    ----------
    request : SequenceRepresentationRequest
        The request object containing all parameters for representation extraction.
        Must have the following attributes:
        - model_name (str): Name of the model to use for representation extraction
        - model_type (str): Type of model (e.g., 'transformer', 'cnn', 'lstm')
        - sequences (List[str]): List of biological sequences to process
        - representation_type (str): Type of representation to extract
          ('cls', 'pooled', or 'full')
    model_manager : ModelManager
        The model manager instance responsible for loading and managing models.
        Used to retrieve the specified model for representation extraction.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the representation results with the following keys:
        - embeddings (List[List[float]] or List[List[List[float]]]): Extracted embeddings
          - For 'cls' and 'pooled': List[List[float]] - one embedding per sequence
          - For 'full': List[List[List[float]]] - embeddings for each position in each sequence
        - embedding_dimension (int): Dimensionality of the embeddings
        - num_sequences (int): Number of sequences processed
        - representation_type (str): Type of representation that was extracted
        - model_used (str): Identifier of the model used for extraction

    Raises
    ------
    ValueError
        If an unknown representation_type is specified (not 'cls', 'pooled', or 'full').
    Exception
        If there is an error loading the model or processing the sequences.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure for CLS token representations
    >>> request = SequenceRepresentationRequest(
    ...     model_name="protein_transformer",
    ...     model_type="transformer",
    ...     sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
    ...     representation_type="cls"
    ... )
    >>>
    >>> # The function would be called by the MCP server
    >>> result = get_sequence_representations(request, model_manager)
    >>>
    >>> # Example response structure for CLS representations
    >>> {
    ...     "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
    ...     "embedding_dimension": 10,
    ...     "num_sequences": 1,
    ...     "representation_type": "cls",
    ...     "model_used": "transformer_protein_transformer"
    ... }

    >>> # Example for pooled representations
    >>> request.representation_type = "pooled"
    >>> result = get_sequence_representations(request, model_manager)
    >>> # Returns mean-pooled embeddings across all sequence positions

    >>> # Example for full sequence representations
    >>> request.representation_type = "full"
    >>> result = get_sequence_representations(request, model_manager)
    >>> # Returns embeddings for each position in the sequence

    Notes
    -----
    - The function uses torch.no_grad() for inference to save memory
    - Representation types:
      * 'cls': Uses the CLS token representation (first token) - good for sequence-level tasks
      * 'pooled': Mean pooling across all sequence positions - provides sequence-level summary
      * 'full': All position-wise representations - useful for position-specific analysis
    - All tensors are converted to CPU and then to Python lists for JSON serialization
    - The embedding dimension depends on the model architecture and training
    - This tool is useful for extracting features for downstream machine learning tasks,
      sequence similarity analysis, or understanding model representations
    """
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
