"""Pydantic models for request validation in Lobster MCP server."""

from pydantic import BaseModel, Field


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
