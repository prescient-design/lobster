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
from typing import Any

import torch
from fastmcp import FastMCP
from pydantic import BaseModel, Field

try:
    from lobster.model import LobsterCBMPMLM, LobsterPMLM
except ImportError as e:
    raise ImportError("Please install lobster: pip install -e .") from e

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lobster-fastmcp-server")

# Available pretrained models
AVAILABLE_MODELS = {
    "masked_lm": {
        "lobster_24M": "asalam91/lobster_24M",
        "lobster_150M": "asalam91/lobster_150M",
    },
    "concept_bottleneck": {
        "cb_lobster_24M": "asalam91/cb_lobster_24M",
        "cb_lobster_150M": "asalam91/cb_lobster_150M",
        "cb_lobster_650M": "asalam91/cb_lobster_650M",
        "cb_lobster_3B": "asalam91/cb_lobster_3B",
    },
}


# Pydantic models for input validation
class SequenceRepresentationRequest(BaseModel):
    sequences: list[str] = Field(..., description="List of protein sequences")
    model_name: str = Field(..., description="Name of the model to use")
    model_type: str = Field(..., description="Type of model: 'masked_lm' or 'concept_bottleneck'")
    representation_type: str = Field(default="pooled", description="Type of representation: 'cls', 'pooled', or 'full'")


class SequenceConceptsRequest(BaseModel):
    sequences: list[str] = Field(..., description="List of protein sequences")
    model_name: str = Field(..., description="Name of the concept bottleneck model")


class InterventionRequest(BaseModel):
    sequence: str = Field(..., description="Protein sequence to modify")
    concept: str = Field(..., description="Concept to intervene on")
    model_name: str = Field(..., description="Name of the concept bottleneck model")
    edits: int = Field(default=5, description="Number of edits to make")
    intervention_type: str = Field(default="negative", description="Type of intervention: 'positive' or 'negative'")


class SupportedConceptsRequest(BaseModel):
    model_name: str = Field(..., description="Name of the concept bottleneck model")


class NaturalnessRequest(BaseModel):
    sequences: list[str] = Field(..., description="List of protein sequences")
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model: 'masked_lm' or 'concept_bottleneck'")


class LobsterInferenceServer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_models: dict[str, LobsterPMLM | LobsterCBMPMLM] = {}
        logger.info(f"Initialized Lobster FastMCP Server on device: {self.device}")

    def _get_or_load_model(self, model_name: str, model_type: str) -> LobsterPMLM | LobsterCBMPMLM:
        """Load model if not already cached"""
        cache_key = f"{model_type}_{model_name}"

        if cache_key not in self.loaded_models:
            logger.info(f"Loading model {model_name} of type {model_type}")

            if model_type == "masked_lm":
                if model_name not in AVAILABLE_MODELS["masked_lm"]:
                    raise ValueError(f"Unknown masked LM model: {model_name}")
                model_path = AVAILABLE_MODELS["masked_lm"][model_name]
                model = LobsterPMLM(model_path).to(self.device)

            elif model_type == "concept_bottleneck":
                if model_name not in AVAILABLE_MODELS["concept_bottleneck"]:
                    raise ValueError(f"Unknown concept bottleneck model: {model_name}")
                model_path = AVAILABLE_MODELS["concept_bottleneck"][model_name]
                model = LobsterCBMPMLM(model_path).to(self.device)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model.eval()
            self.loaded_models[cache_key] = model
            logger.info(f"Successfully loaded {cache_key}")

        return self.loaded_models[cache_key]


# Initialize FastMCP server
app = FastMCP("Lobster Inference")
lobster_server = LobsterInferenceServer()


@app.tool()
def list_available_models() -> dict[str, Any]:
    """List all available pretrained Lobster models and current device"""
    return {"available_models": AVAILABLE_MODELS, "device": lobster_server.device}


@app.tool()
def get_sequence_representations(request: SequenceRepresentationRequest) -> dict[str, Any]:
    """Get sequence representations from a model"""
    try:
        model = lobster_server._get_or_load_model(request.model_name, request.model_type)

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


@app.tool()
def get_sequence_concepts(request: SequenceConceptsRequest) -> dict[str, Any]:
    """Get concept predictions from a concept bottleneck model"""
    try:
        model = lobster_server._get_or_load_model(request.model_name, "concept_bottleneck")

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


@app.tool()
def intervene_on_sequence(request: InterventionRequest) -> dict[str, Any]:
    """Perform concept intervention on a sequence"""
    try:
        model = lobster_server._get_or_load_model(request.model_name, "concept_bottleneck")

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


@app.tool()
def get_supported_concepts(request: SupportedConceptsRequest) -> dict[str, Any]:
    """Get list of supported concepts for a concept bottleneck model"""
    try:
        model = lobster_server._get_or_load_model(request.model_name, "concept_bottleneck")
        concepts = model.list_supported_concept()

        return {
            "supported_concepts": concepts,
            "num_concepts": len(concepts) if isinstance(concepts, list) else None,
            "model_used": f"concept_bottleneck_{request.model_name}",
        }

    except Exception as e:
        logger.error(f"Error getting supported concepts: {str(e)}")
        raise


@app.tool()
def compute_naturalness(request: NaturalnessRequest) -> dict[str, Any]:
    """Compute naturalness/likelihood scores for sequences"""
    try:
        model = lobster_server._get_or_load_model(request.model_name, request.model_type)

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


def main():
    """Main entry point for FastMCP server"""
    app.run()


if __name__ == "__main__":
    main()
