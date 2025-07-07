"""Tests for the DGEB adapter."""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from lobster.evaluation.dgeb_adapter import UMEAdapterDGEB
from lobster.model import UME

logger = logging.getLogger(__name__)


def check_aws_credentials():
    """Check if AWS credentials are available."""
    # Check environment variables
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True

    # Check AWS credentials file
    aws_creds_path = Path.home() / ".aws" / "credentials"
    if aws_creds_path.exists():
        return True

    # Check if we're in an environment with IAM roles (like EC2)
    try:
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials and credentials.access_key:
            return True
    except Exception:
        pass

    return False


@pytest.fixture
def test_model_checkpoint():
    """Create a small UME model checkpoint for testing."""
    # Create a minimal UME model
    model = UME(
        model_name="UME_mini",  # Use the smallest variant
        max_length=128,  # Short sequences for testing
        use_flash_attn=False,  # CPU-compatible
    )

    # Save the model to a temporary checkpoint
    temp_dir = Path(tempfile.mkdtemp())
    checkpoint_path = temp_dir / "test_ume_model.ckpt"

    # Save the model directly using torch.save (simpler approach)
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": model.hparams,
        "epoch": 0,
        "global_step": 0,
        "pytorch-lightning_version": "2.0.0",
        "state_dict_type": "model",
    }
    torch.save(checkpoint, checkpoint_path)

    yield str(checkpoint_path)

    # Cleanup
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()


def test_layer_extraction_different_embeddings(test_model_checkpoint):
    """Test that different layers produce different embeddings."""

    # Initialize the adapter with the test model checkpoint
    adapter = UMEAdapterDGEB(
        model_name=test_model_checkpoint,
        layers=[0, 2, 5],  # Test specific layers
        modality="protein",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,  # CPU-compatible
    )

    # Test sequences
    test_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

    # Extract embeddings
    embeddings = adapter.encode(test_sequences)

    # Check shape - should match the number of requested layers
    expected_shape = (len(test_sequences), len(adapter.layers), adapter.embed_dim)
    assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"

    # Check that embeddings are not all zeros
    assert not np.allclose(embeddings, 0), "All embeddings are zero!"

    # Check that embeddings are different across layers
    layer_embeddings = embeddings[0]  # First sequence, all layers

    # Test that at least some layers are different
    different_layers_found = False

    for i in range(len(adapter.layers)):
        for j in range(i + 1, len(adapter.layers)):
            layer_i = layer_embeddings[i]
            layer_j = layer_embeddings[j]

            # Calculate cosine similarity
            cos_sim = np.dot(layer_i, layer_j) / (np.linalg.norm(layer_i) * np.linalg.norm(layer_j))

            # Check that layers are different (cosine similarity < 0.99)
            if cos_sim < 0.99:
                different_layers_found = True
                break

        if different_layers_found:
            break

    assert different_layers_found, "All layers produce identical embeddings!"


def test_layer_extraction_mid_last_different(test_model_checkpoint):
    """Test that 'mid' and 'last' layer specifications produce different results."""

    # Test with 'mid' layers
    adapter_mid = UMEAdapterDGEB(
        model_name=test_model_checkpoint,
        layers="mid",
        modality="protein",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,  # CPU-compatible
    )

    # Test with 'last' layers
    adapter_last = UMEAdapterDGEB(
        model_name=test_model_checkpoint,
        layers="last",
        modality="protein",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,  # CPU-compatible
    )

    # Test sequences
    test_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ]

    # Extract embeddings
    embeddings_mid = adapter_mid.encode(test_sequences)
    embeddings_last = adapter_last.encode(test_sequences)

    # Both should have 1 layer each (mid and last both select 1 layer)
    assert embeddings_mid.shape[1] == 1, f"Expected 1 layer for mid, got {embeddings_mid.shape[1]}"
    assert embeddings_last.shape[1] == 1, f"Expected 1 layer for last, got {embeddings_last.shape[1]}"

    # Check that the embeddings are different
    # For 'mid', we get middle layers; for 'last', we get the last layer
    # They should be different embeddings
    mid_embedding = embeddings_mid[0, 0]  # First sequence, first layer
    last_embedding = embeddings_last[0, 0]  # First sequence, first layer

    # Calculate cosine similarity
    cos_sim = np.dot(mid_embedding, last_embedding) / (np.linalg.norm(mid_embedding) * np.linalg.norm(last_embedding))

    # They should be different (cosine similarity < 0.99)
    assert cos_sim < 0.99, f"Mid and last layer embeddings are too similar (cos_sim={cos_sim:.6f})"


def test_layer_extraction_specific_layers(test_model_checkpoint):
    """Test that specific layer indices produce different embeddings."""

    # Test with specific layer indices
    adapter = UMEAdapterDGEB(
        model_name=test_model_checkpoint,
        layers=[0, 3, 5],  # Test specific layers
        modality="protein",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,  # CPU-compatible
    )

    # Test sequences
    test_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    ]

    # Extract embeddings
    embeddings = adapter.encode(test_sequences)

    # Should have 3 layers (as specified)
    assert embeddings.shape[1] == 3, f"Expected 3 layers, got {embeddings.shape[1]}"

    # Check that the layers are different
    layer_0 = embeddings[0, 0]
    layer_3 = embeddings[0, 1]  # This should be layer 3 from the model
    layer_5 = embeddings[0, 2]  # This should be layer 5 from the model

    # Calculate similarities
    cos_sim_0_3 = np.dot(layer_0, layer_3) / (np.linalg.norm(layer_0) * np.linalg.norm(layer_3))
    cos_sim_0_5 = np.dot(layer_0, layer_5) / (np.linalg.norm(layer_0) * np.linalg.norm(layer_5))
    cos_sim_3_5 = np.dot(layer_3, layer_5) / (np.linalg.norm(layer_3) * np.linalg.norm(layer_5))

    # All should be different
    assert cos_sim_0_3 < 0.99, f"Layers 0 and 3 are too similar (cos_sim={cos_sim_0_3:.6f})"
    assert cos_sim_0_5 < 0.99, f"Layers 0 and 5 are too similar (cos_sim={cos_sim_0_5:.6f})"
    assert cos_sim_3_5 < 0.99, f"Layers 3 and 5 are too similar (cos_sim={cos_sim_3_5:.6f})"


if __name__ == "__main__":
    # Run tests
    test_layer_extraction_different_embeddings()
    test_layer_extraction_mid_last_different()
    test_layer_extraction_specific_layers()
    print("🎉 All tests passed! The DGEB adapter now properly extracts different layer embeddings.")
