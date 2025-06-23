from unittest.mock import MagicMock

import pytest
import torch

from lobster.model import UME
from lobster.server._server import UMEServer


@pytest.fixture
def sample_sequences():
    return {
        "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"],
        "nucleotide": ["ATGCATGC", "GCTAGCTA"],
        "3d_coordinates": [["aa", "bb", "cc", "dd"], ["aa", "bb", "cc", "dd"]],
    }


@pytest.fixture
def mock_ume_model():
    model = MagicMock(spec=UME)

    # Mock the embed_sequences method to return dummy tensors
    def mock_embed_sequences(sequences, modality, aggregate=True):
        batch_size = len(sequences) if isinstance(sequences, list) else 1
        if aggregate:
            return torch.randn(batch_size, 128)  # [batch_size, hidden_size]
        else:
            return torch.randn(batch_size, 10, 128)  # [batch_size, seq_length, hidden_size]

    model.embed_sequences.side_effect = mock_embed_sequences
    return model


@pytest.fixture
def ume_server(mock_ume_model):
    return UMEServer(mock_ume_model)


def test_ume_mcp_server(ume_server, sample_sequences):
    # Test each modality
    for modality, sequences in sample_sequences.items():
        # Get embeddings through MCP server
        embeddings = ume_server.embed_sequences(sequences, modality, aggregate=False)

        # Verify embeddings shape and type
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 3  # [batch_size, seq_length, hidden_size]
        assert embeddings.shape[0] == len(sequences)
        assert embeddings.shape[1] == 10  # Mock sequence length

        # Test with aggregation
        embeddings_agg = ume_server.embed_sequences(sequences, modality, aggregate=True)
        assert isinstance(embeddings_agg, torch.Tensor)
        assert embeddings_agg.dim() == 2  # [batch_size, hidden_size]
        assert embeddings_agg.shape[0] == len(sequences)

        # Verify that mock was called correctly
        ume_server.model.embed_sequences.assert_called_with(sequences, modality, aggregate=True)


def test_ume_mcp_server_single_sequence(ume_server):
    # Test with single sequence
    sequence = "CC(=O)OC1=CC=CC=C1C(=O)O"
    embeddings = ume_server.embed_sequences(sequence, "SMILES")

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2  # [1, hidden_size] since aggregate=True by default
    assert embeddings.shape[0] == 1

    # Verify that mock was called correctly
    ume_server.model.embed_sequences.assert_called_with(sequence, "SMILES", aggregate=True)


def test_ume_mcp_server_invalid_modality(ume_server):
    # Configure mock to raise ValueError for invalid modality
    ume_server.model.embed_sequences.side_effect = ValueError("Invalid modality")

    # Test with invalid modality
    with pytest.raises(ValueError):
        ume_server.embed_sequences(["test"], "invalid_modality")

    # Verify that mock was called correctly
    ume_server.model.embed_sequences.assert_called_with(["test"], "invalid_modality", aggregate=True)
