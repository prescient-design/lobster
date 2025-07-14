#!/usr/bin/env python3
"""
Test script for UME DGEB implementation.

This script performs basic tests to ensure the UME DGEB protein wrapper works correctly.
"""

import torch
import numpy as np
from unittest.mock import Mock, patch

from lobster.model._ume_dgeb import (
    UMEProteinTransformer,
    create_ume_protein_transformer,
)


class MockUME:
    """Mock UME model for testing."""

    def __init__(self, *args, **kwargs):
        self.embedding_dim = 768
        self.model = Mock()
        self.model.config = Mock()
        self.model.config.num_hidden_layers = 12
        self.model.config.hidden_size = 768
        self.device = torch.device("cpu")

    def freeze(self):
        pass

    def get_tokenizer(self, modality):
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.mask_token_id = 1
        tokenizer.cls_token_id = 2
        tokenizer.eos_token_id = 3
        return tokenizer

    def embed_sequences(self, sequences, modality, aggregate=True):
        batch_size = len(sequences)
        if aggregate:
            return torch.randn(batch_size, self.embedding_dim)
        else:
            return torch.randn(batch_size, 100, self.embedding_dim)  # Mock seq_len = 100

    def to(self, device):
        self.device = device
        return self


@patch("lobster.model._ume_dgeb.UME")
def test_protein_transformer_initialization(mock_ume_class):
    """Test UME protein transformer initialization."""
    mock_ume_class.from_pretrained.return_value = MockUME()

    transformer = UMEProteinTransformer("test-model")

    # Check properties
    assert transformer.embed_dim == 768
    assert transformer.num_layers == 12
    assert hasattr(transformer, "modality")


@patch("lobster.model._ume_dgeb.UME")
def test_protein_transformer_encoding(mock_ume_class):
    """Test protein sequence encoding."""
    mock_ume_class.from_pretrained.return_value = MockUME()

    transformer = create_ume_protein_transformer("test-model")

    # Test with protein sequences
    sequences = ["MKTII", "AELEL", "GKVLW"]
    embeddings = transformer.encode(sequences)

    # Check output format
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 3  # Number of sequences
    assert embeddings.shape[2] == 768  # Embedding dimension


@patch("lobster.model._ume_dgeb.UME")
def test_empty_sequences(mock_ume_class):
    """Test handling of empty sequence lists."""
    mock_ume_class.from_pretrained.return_value = MockUME()

    transformer = create_ume_protein_transformer("test-model")

    # Test with empty list
    embeddings = transformer.encode([])
    assert embeddings.shape[0] == 0


@patch("lobster.model._ume_dgeb.UME")
def test_single_sequence(mock_ume_class):
    """Test handling of single sequence."""
    mock_ume_class.from_pretrained.return_value = MockUME()

    transformer = create_ume_protein_transformer("test-model")

    # Test with single sequence
    embeddings = transformer.encode(["MKTII"])
    assert embeddings.shape[0] == 1


@patch("lobster.model._ume_dgeb.UME")
def test_l2_normalization(mock_ume_class):
    """Test L2 normalization option."""
    mock_ume = MockUME()
    mock_ume_class.from_pretrained.return_value = mock_ume

    transformer = create_ume_protein_transformer("test-model", l2_norm=True)

    sequences = ["MKTII", "AELEL"]
    embeddings = transformer.encode(sequences)

    # Check that embeddings are normalized (approximately)
    # Note: This test might need adjustment based on actual normalization implementation
    norms = np.linalg.norm(embeddings, axis=-1)
    # With L2 normalization, norms should be close to 1
    assert np.allclose(norms, 1.0, atol=1e-6)


@patch("lobster.model._ume_dgeb.UME")
def test_convenience_function(mock_ume_class):
    """Test convenience function for creating protein transformer."""
    mock_ume_class.from_pretrained.return_value = MockUME()

    transformer = create_ume_protein_transformer("test-model")

    assert isinstance(transformer, UMEProteinTransformer)


@patch("lobster.model._ume_dgeb.UME")
def test_checkpoint_loading(mock_ume_class):
    """Test loading from checkpoint."""
    mock_ume_class.load_from_checkpoint.return_value = MockUME()

    # Test loading from checkpoint path
    UMEProteinTransformer(model_name="path/to/checkpoint.ckpt", use_pretrained=True)

    # Should call load_from_checkpoint instead of from_pretrained
    mock_ume_class.load_from_checkpoint.assert_called_once()


@patch("lobster.model._ume_dgeb.UME")
def test_custom_parameters(mock_ume_class):
    """Test custom parameters are passed correctly."""
    mock_ume_class.from_pretrained.return_value = MockUME()

    transformer = create_ume_protein_transformer(
        model_name="test-model",
        max_seq_length=512,
        batch_size=16,
        l2_norm=True,
        use_flash_attn=False,
    )

    # Check that parameters are stored
    assert transformer.max_seq_length == 512
    assert transformer.batch_size == 16
    assert transformer.l2_norm is True


def run_basic_test():
    """Run a basic test manually without pytest."""
    print("Running basic UME DGEB test...")

    with patch("lobster.model._ume_dgeb.UME") as mock_ume:
        mock_ume.from_pretrained.return_value = MockUME()

        # Test basic functionality
        transformer = create_ume_protein_transformer("test-model")
        sequences = ["MKTII", "AELEL"]
        embeddings = transformer.encode(sequences)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        print("✓ Basic functionality test passed")

        # Test with empty sequences
        empty_embeddings = transformer.encode([])
        assert empty_embeddings.shape[0] == 0
        print("✓ Empty sequences test passed")

        # Test single sequence
        single_embeddings = transformer.encode(["MKTII"])
        assert single_embeddings.shape[0] == 1
        print("✓ Single sequence test passed")
