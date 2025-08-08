"""Tests for the ESM DGEB adapter - focused on core encoding functionality."""

import logging

import numpy as np
import pytest
import torch
import lightning as L

from lobster.evaluation.esm_dgeb_adapter import ESMAdapterDGEB

logger = logging.getLogger(__name__)


class MockESMModule(L.LightningModule):
    """Mock ESM module for testing."""

    def __init__(self, embed_dim=1280):
        super().__init__()
        self.embed_dim = embed_dim
        self.dummy_param = torch.nn.Parameter(torch.randn(embed_dim, embed_dim))

    def embed_sequences(self, sequences, modality="amino_acid", aggregate=True):
        """Mock embed_sequences method."""
        if aggregate:
            # Return aggregated embeddings (old behavior)
            embeddings = []
            for i, seq in enumerate(sequences):
                # Create deterministic embeddings based on sequence length and index
                embedding = torch.randn(self.embed_dim) + (len(seq) / 100.0) + (i * 0.1)
                embeddings.append(embedding)
            return torch.stack(embeddings)
        else:
            # Return token-level embeddings (new behavior)
            token_embeddings = []
            for i, seq in enumerate(sequences):
                seq_len = len(seq) + 2  # Add special tokens
                # Create token-level embeddings with some padding at the end
                token_emb = torch.randn(seq_len, self.embed_dim) + (i * 0.1)
                # Simulate padding tokens as zeros at the end
                token_emb[-1:] = 0.0  # Last token is padding
                token_embeddings.append(token_emb)

            # Return as list (variable length) if sequences have different lengths
            if len(set(len(seq) for seq in sequences)) > 1:
                return token_embeddings
            else:
                # All same length, can stack into tensor
                return torch.stack(token_embeddings)

    def parameters(self):
        yield self.dummy_param


def mock_process_and_embed(module, sequences, modality="amino_acid", aggregate=True):
    """Mock processing function that calls the module's embed_sequences method."""
    return module.embed_sequences(sequences, modality, aggregate)


def create_adapter(esm_module, **kwargs):
    """Helper function to create adapter with common defaults."""
    defaults = {
        "module": esm_module,
        "modality": "protein",
        "batch_size": 32,
        "max_seq_length": 1024,
        "l2_norm": False,
        "process_and_embed_fn": mock_process_and_embed,
    }
    defaults.update(kwargs)
    return ESMAdapterDGEB(**defaults)


@pytest.fixture
def mock_esm_module():
    """Create a mock ESM module for testing."""
    return MockESMModule(embed_dim=1280)


@pytest.fixture
def adapter(mock_esm_module):
    """Create a basic adapter for testing."""
    return create_adapter(mock_esm_module)


class TestESMAdapterDGEB:
    """Test suite for ESMAdapterDGEB - focused on core encoding functionality."""

    def test_adapter_initialization_and_basic_properties(self, mock_esm_module):
        """Test basic adapter initialization and core properties."""
        adapter = create_adapter(mock_esm_module, batch_size=16, l2_norm=True)

        assert adapter.esm_module is mock_esm_module
        assert adapter.embed_dim == 1280
        assert adapter.num_layers == 1
        assert adapter.batch_size == 16
        assert adapter.l2_norm is True

        # Test that config is added to module
        assert hasattr(mock_esm_module, "config")
        assert mock_esm_module.config.hidden_size == 1280

    def test_initialization_requires_process_function(self, mock_esm_module):
        """Test that process_and_embed_fn is required."""
        with pytest.raises(ValueError, match="process_and_embed_fn must be provided"):
            ESMAdapterDGEB(
                module=mock_esm_module,
                modality="protein",
                process_and_embed_fn=None,
            )

    def test_basic_encoding(self, adapter):
        """Test basic sequence encoding functionality."""
        test_sequences = ["MKTVRQERLK", "ARNDCEQGHILKMFPSTWYVX", "MGKIHR"]

        embeddings = adapter.encode(test_sequences)

        # Check shape: [num_sequences, num_layers, embedding_dim]
        expected_shape = (3, 1, 1280)
        assert embeddings.shape == expected_shape

        # Check that embeddings are not all zeros
        assert not np.allclose(embeddings, 0)

        # Check that different sequences produce different embeddings
        seq1_emb = embeddings[0, 0, :]
        seq2_emb = embeddings[1, 0, :]
        assert not np.allclose(seq1_emb, seq2_emb, atol=1e-6)

    def test_encoding_with_l2_normalization(self, mock_esm_module):
        """Test encoding with L2 normalization."""
        adapter = create_adapter(mock_esm_module, l2_norm=True)
        test_sequences = ["MKTVRQERLK", "ARNDCEQGHILKMFPSTWYVX"]
        embeddings = adapter.encode(test_sequences)

        # Check that embeddings are normalized (unit norm)
        norms = np.linalg.norm(embeddings, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encoding_edge_cases(self, adapter):
        """Test encoding edge cases: empty lists and invalid strings."""
        # Empty list
        assert adapter.encode([]).size == 0

        # Empty/whitespace strings
        invalid_sequences = ["", "   ", "\t", "\n"]
        assert adapter.encode(invalid_sequences).size == 0

    def test_encoding_batch_processing(self, mock_esm_module):
        """Test that sequences are processed in batches correctly."""
        adapter = create_adapter(mock_esm_module, batch_size=2)

        # More sequences than batch size
        test_sequences = ["MKTVRQERLK", "ARNDCEQGHILKMFPSTWYVX", "MGKIHR", "ALANYLCYS", "GLYHISPRO"]

        embeddings = adapter.encode(test_sequences)

        # Should produce correct shape regardless of batching
        assert embeddings.shape == (5, 1, 1280)

    def test_encoding_error_handling(self, mock_esm_module):
        """Test error handling during encoding."""

        def failing_process_and_embed(module, sequences, modality, aggregate):
            if len(sequences) > 1:
                raise RuntimeError("Simulated encoding error")
            return module.embed_sequences(sequences, modality, aggregate)

        adapter = create_adapter(mock_esm_module, batch_size=2, process_and_embed_fn=failing_process_and_embed)

        # This should trigger error handling (batch size > 1)
        test_sequences = ["MKTVRQERLK", "ARNDCEQGHILKMFPSTWYVX"]
        embeddings = adapter.encode(test_sequences)

        # Should return zeros for failed batches
        assert embeddings.shape == (2, 1, 1280)
        assert np.allclose(embeddings, 0)

    def test_different_modalities(self, mock_esm_module):
        """Test adapter with different modalities."""
        from dgeb.modality import Modality

        # Test protein modality
        protein_adapter = create_adapter(mock_esm_module, modality="protein")
        assert protein_adapter.modality == Modality.PROTEIN

        # Test DNA modality
        dna_adapter = create_adapter(mock_esm_module, modality="dna")
        assert dna_adapter.modality == Modality.DNA

        # Test invalid modality
        invalid_adapter = create_adapter(mock_esm_module, modality="invalid")
        with pytest.raises(ValueError, match="Unsupported modality: invalid"):
            _ = invalid_adapter.modality

    def test_different_embedding_dimensions(self, mock_esm_module):
        """Test adapter with different embedding dimensions."""
        small_module = MockESMModule(embed_dim=640)
        adapter = create_adapter(small_module)

        assert adapter.embed_dim == 640

        test_sequences = ["MKTVRQERLK", "ARNDCEQGHILKMFPSTWYVX"]
        embeddings = adapter.encode(test_sequences)

        assert embeddings.shape == (2, 1, 640)
