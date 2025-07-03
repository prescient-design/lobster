"""Test DGEB integration with UME models."""

import logging
import os
import tempfile
from pathlib import Path

import pytest
import torch

from lobster.evaluation.dgeb_adapter import UMEAdapter
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


def test_dgeb_tasks():
    """Test that DGEB tasks can be loaded."""
    import dgeb

    # Get available tasks
    all_tasks = dgeb.get_all_task_names()
    assert len(all_tasks) > 0, "Should find some tasks"

    # Get protein tasks
    protein_tasks = dgeb.get_tasks_by_modality(dgeb.modality.Modality.PROTEIN)
    assert len(protein_tasks) > 0, "Should find protein tasks"

    # Get DNA tasks
    dna_tasks = dgeb.get_tasks_by_modality(dgeb.modality.Modality.DNA)
    assert len(dna_tasks) > 0, "Should find DNA tasks"


@pytest.mark.skipif(not check_aws_credentials(), reason="AWS credentials not available for pretrained model testing")
def test_ume_adapter_pretrained():
    """Test UMEAdapter with pretrained model (requires AWS credentials)."""
    adapter = UMEAdapter(
        model_name="ume-mini-base-12M",
        modality="protein",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,  # Use CPU-compatible settings
    )

    assert adapter.embed_dim > 0
    assert adapter.num_layers > 0  # Should be actual number of layers from UME model
    assert adapter.modality.value == "protein"

    # Test encoding
    test_sequences = [
        "MKLLNVINFVFLMFVSSSKILGMTAFPPPPTSLFSKEMHLDGLPTQRWLRQNTHPLLKIAPCFHVDQFPQKSGKLEFMNLFRSAPPHPSSQIGSMVFHVKLICGQHRVLDGLWPTLLTDGFDPGGR",
        "MVKFLPHLGNEKNTLCAIVRYTKQSEQNLVSSVIGFNLLPHVCRTVEAKYLTRLVSVAEKCSFTVEGVKNSTGTLGAATLSTHKLLKSLLVFFAYIVV",
    ]

    embeddings = adapter.encode(test_sequences)
    expected_shape = (len(test_sequences), adapter.num_layers, adapter.embed_dim)
    assert embeddings.shape == expected_shape


def test_ume_adapter_from_checkpoint(test_model_checkpoint):
    """Test UMEAdapter with model created from scratch."""
    adapter = UMEAdapter(
        model_name=test_model_checkpoint,
        modality="protein",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,
    )

    assert adapter.embed_dim > 0
    assert adapter.num_layers > 0  # Should be actual number of layers from UME model
    assert adapter.modality.value == "protein"

    # Test encoding
    test_sequences = [
        "MKLLNVINFVFLMFVSSSKILGMTAFPPPPTSLFSKEMHLDGLPTQRWLRQNTHPLLKIAPCFHVDQFPQKSGKLEFMNLFRSAPPHPSSQIGSMVFHVKLICGQHRVLDGLWPTLLTDGFDPGGR",
        "MVKFLPHLGNEKNTLCAIVRYTKQSEQNLVSSVIGFNLLPHVCRTVEAKYLTRLVSVAEKCSFTVEGVKNSTGTLGAATLSTHKLLKSLLVFFAYIVV",
    ]

    embeddings = adapter.encode(test_sequences)
    expected_shape = (len(test_sequences), adapter.num_layers, adapter.embed_dim)
    assert embeddings.shape == expected_shape


def test_ume_adapter_empty_sequences(test_model_checkpoint):
    """Test UMEAdapter with empty sequences."""
    adapter = UMEAdapter(
        model_name=test_model_checkpoint,
        modality="protein",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,
    )

    # Test with empty list
    empty_result = adapter.encode([])
    assert len(empty_result) == 0

    # Test with invalid sequences
    invalid_result = adapter.encode(["", "   ", "\t\n"])
    assert len(invalid_result) == 0


def test_ume_adapter_dna_modality(test_model_checkpoint):
    """Test UMEAdapter with DNA modality."""
    adapter = UMEAdapter(
        model_name=test_model_checkpoint,
        modality="dna",
        batch_size=2,
        max_seq_length=128,
        use_flash_attn=False,
    )

    assert adapter.modality.value == "dna"

    # Test encoding DNA sequences
    dna_sequences = [
        "ATGCATGCATGCATGCATGC",
        "GCTAGCTAGCTAGCTAGCTA",
    ]

    embeddings = adapter.encode(dna_sequences)
    expected_shape = (len(dna_sequences), adapter.num_layers, adapter.embed_dim)
    assert embeddings.shape == expected_shape


def test_ume_adapter_metadata(test_model_checkpoint):
    """Test UMEAdapter metadata property."""
    adapter = UMEAdapter(
        model_name=test_model_checkpoint,
        modality="protein",
        batch_size=32,
        max_seq_length=256,
        use_flash_attn=False,
        l2_norm=True,
        pool_type="mean",
    )

    metadata = adapter.metadata
    assert metadata["modality"] == "protein"
    assert metadata["batch_size"] == 32
    assert metadata["max_seq_length"] == 256
    assert metadata["l2_norm"] is True
    assert metadata["pool_type"] == "mean"
    assert "embed_dim" in metadata
    assert metadata["embed_dim"] > 0
