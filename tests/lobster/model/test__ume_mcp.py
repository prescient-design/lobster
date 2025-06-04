import pytest
import torch
from mcp.server.fastmcp import FastMCP

from lobster.model import Ume


@pytest.fixture
def sample_sequences():
    return {
        "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"],
        "nucleotide": ["ATGCATGC", "GCTAGCTA"],
        "3d_coordinates": [["aa", "bb", "cc", "dd"], ["aa", "bb", "cc", "dd"]],
    }


@pytest.fixture
def ume_model():
    return Ume(model_name="UME_mini", max_length=10)


def test_ume_mcp_server(ume_model, sample_sequences):
    # Create FastMCP server
    server = FastMCP()

    # Register the embed_sequences method
    @server.tool(description="Embed sequences into a latent space")
    def embed_sequences(sequences, modality, aggregate=True):
        return ume_model.embed_sequences(sequences, modality, aggregate=aggregate)

    # Test each modality
    for modality, sequences in sample_sequences.items():
        # Get embeddings through MCP server
        embeddings = embed_sequences(sequences, modality, aggregate=False)

        # Verify embeddings shape and type
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 3  # [batch_size, seq_length, hidden_size]
        assert embeddings.shape[0] == len(sequences)
        assert embeddings.shape[1] <= ume_model.max_length

        # Test with aggregation
        embeddings_agg = embed_sequences(sequences, modality, aggregate=True)
        assert isinstance(embeddings_agg, torch.Tensor)
        assert embeddings_agg.dim() == 2  # [batch_size, hidden_size]
        assert embeddings_agg.shape[0] == len(sequences)

        # Verify that aggregated embeddings match mean of non-aggregated
        mean_embeddings = embeddings.mean(dim=1)
        torch.testing.assert_close(embeddings_agg, mean_embeddings, rtol=1e-5, atol=1e-5)


def test_ume_mcp_server_single_sequence(ume_model):
    server = FastMCP()

    @server.tool(description="Embed sequences into a latent space")
    def embed_sequences(sequences, modality, aggregate=True):
        return ume_model.embed_sequences(sequences, modality, aggregate=aggregate)

    # Test with single sequence
    sequence = "CC(=O)OC1=CC=CC=C1C(=O)O"
    embeddings = embed_sequences(sequence, "SMILES")

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2  # [1, hidden_size] since aggregate=True by default
    assert embeddings.shape[0] == 1


def test_ume_mcp_server_invalid_modality(ume_model):
    server = FastMCP()

    @server.tool(description="Embed sequences into a latent space")
    def embed_sequences(sequences, modality, aggregate=True):
        return ume_model.embed_sequences(sequences, modality, aggregate=aggregate)

    # Test with invalid modality
    with pytest.raises(ValueError):
        embed_sequences(["test"], "invalid_modality")
