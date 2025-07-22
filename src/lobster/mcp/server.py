#!/usr/bin/env python3
"""
FastMCP Server for Lobster Model Inference and Interventions

This server provides access to pretrained Lobster models for:
1. Getting sequence representations (embeddings)
2. Performing concept interventions on sequences
3. Computing sequence likelihoods/naturalness

Available models:
- LobsterPMLM: Masked Language Models (24M, 150M)
- LobsterCBMPMLM: Concept Bottleneck Models (24M, 150M, 650M, 3B)
"""

import logging

from fastmcp import FastMCP

from .models import ModelManager
from .schemas import (
    InterventionRequest,
    NaturalnessRequest,
    SequenceConceptsRequest,
    SequenceRepresentationRequest,
    SupportedConceptsRequest,
)
from .tools import (
    compute_naturalness,
    get_sequence_concepts,
    get_sequence_representations,
    get_supported_concepts,
    intervene_on_sequence,
    list_available_models,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lobster-fastmcp-server")

# Initialize FastMCP server and model manager
app = FastMCP("Lobster Inference")
model_manager = ModelManager()


@app.tool()
def list_models():
    """List all available pretrained Lobster models and current device."""
    return list_available_models(model_manager)


@app.tool()
def get_representations(request: SequenceRepresentationRequest):
    """Get sequence representations from a model."""
    return get_sequence_representations(request, model_manager)


@app.tool()
def get_concepts(request: SequenceConceptsRequest):
    """Get concept predictions from a concept bottleneck model."""
    return get_sequence_concepts(request, model_manager)


@app.tool()
def intervene_sequence(request: InterventionRequest):
    """Perform concept intervention on a sequence."""
    return intervene_on_sequence(request, model_manager)


@app.tool()
def get_supported_concepts_list(request: SupportedConceptsRequest):
    """Get list of supported concepts for a concept bottleneck model."""
    return get_supported_concepts(request, model_manager)


@app.tool()
def compute_sequence_naturalness(request: NaturalnessRequest):
    """Compute naturalness/likelihood scores for sequences."""
    return compute_naturalness(request, model_manager)


def main():
    """Main entry point for FastMCP server."""
    app.run()


if __name__ == "__main__":
    main()
