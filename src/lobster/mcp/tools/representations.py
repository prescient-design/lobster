"""Sequence representation tools for Lobster MCP server using FastMCP best practices."""

import logging
from dataclasses import dataclass
from typing import Literal

import torch

from .tool_utils import _load_model

logger = logging.getLogger("lobster-fastmcp-server")


@dataclass
class SequenceRepresentationResult:
    """Result of sequence representation extraction."""

    embeddings: list[list[float]] | list[list[list[float]]]
    embedding_dimension: int
    num_sequences: int
    representation_type: str
    model_used: str


def get_sequence_representations(
    model_name: str,
    sequences: list[str],
    model_type: Literal["masked_lm", "concept_bottleneck"],
    representation_type: Literal["cls", "pooled", "full"] = "pooled",
) -> SequenceRepresentationResult:
    """Get sequence representations from a model.

    This function extracts embeddings from biological sequences using either
    a masked language model or a concept bottleneck model. The embeddings
    can be extracted in different formats depending on the representation_type.

    Parameters
    ----------
    model_name : str
        Name of the model to use for extraction
    sequences : List[str]
        List of biological sequences to process
    model_type : str
        Type of model: 'masked_lm' or 'concept_bottleneck'
    representation_type : str, default="pooled"
        Type of representation to extract:
        - 'cls': Use the [CLS] token representation
        - 'pooled': Use mean pooling over all tokens
        - 'full': Return embeddings for all positions

    Returns
    -------
    SequenceRepresentationResult
        A dataclass containing the representation results with the following fields:
        - embeddings: Extracted embeddings (format depends on representation_type)
        - embedding_dimension: Dimensionality of the embeddings
        - num_sequences: Number of sequences processed
        - representation_type: Type of representation that was extracted
        - model_used: Identifier of the model used for extraction

    Raises
    ------
    Exception
        If there is an error loading the model or processing the sequences.
        The specific error message is logged before raising.

    Examples
    --------
    For MCP server usage, this function is typically called through the MCP protocol:

    >>> # Example MCP request structure
    >>> result = get_sequence_representations(
    ...     model_name="lobster_24M",
    ...     sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
    ...     model_type="masked_lm",
    ...     representation_type="pooled"
    ... )
    >>>
    >>> # Example response structure
    >>> {
    ...     "embeddings": [[0.1, 0.2, 0.3, ...]],
    ...     "embedding_dimension": 512,
    ...     "num_sequences": 1,
    ...     "representation_type": "pooled",
    ...     "model_used": "masked_lm_lobster_24M"
    ... }

    Notes
    -----
    - The function uses torch.no_grad() for inference to save memory
    - For 'cls' and 'pooled' representations, embeddings are returned as List[List[float]]
    - For 'full' representations, embeddings are returned as List[List[List[float]]]
    - All tensors are converted to CPU and then to Python lists for JSON serialization
    - The embedding dimension depends on the specific model architecture
    """
    try:
        model = _load_model(model_name, model_type)

        with torch.no_grad():
            # Get latent representations
            representations = model.sequences_to_latents(sequences)[-1]  # Last layer

            if representation_type == "cls":
                # CLS token representation
                embeddings = representations[:, 0, :]
            elif representation_type == "pooled":
                # Mean pooled representation
                embeddings = torch.mean(representations, dim=1)
            elif representation_type == "full":
                # Full sequence representations
                embeddings = representations
            else:
                raise ValueError(f"Unknown representation type: {representation_type}")

            return SequenceRepresentationResult(
                embeddings=embeddings.cpu().numpy().tolist(),
                embedding_dimension=embeddings.shape[-1],
                num_sequences=len(sequences),
                representation_type=representation_type,
                model_used=f"{model_type}_{model_name}",
            )

    except Exception as e:
        logger.error(f"Error getting representations: {str(e)}")
        raise
