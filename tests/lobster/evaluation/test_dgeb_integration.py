"""Test DGEB integration with UME models."""

import logging
import os
import tempfile
from pathlib import Path
import types

import pytest
import torch

from lobster.evaluation.dgeb_adapter import UMEAdapterDGEB
from lobster.model import UME
from lobster.evaluation import dgeb_runner

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
    """Test UMEAdapterDGEB with pretrained model (requires AWS credentials)."""
    adapter = UMEAdapterDGEB(
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
    expected_shape = (len(test_sequences), 2, adapter.embed_dim)  # 2 is the number of layers the DGEB will use
    assert embeddings.shape == expected_shape


def test_ume_adapter_from_checkpoint(test_model_checkpoint):
    """Test UMEAdapterDGEB with model created from scratch."""
    adapter = UMEAdapterDGEB(
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
    expected_shape = (len(test_sequences), 2, adapter.embed_dim)  # 2 is the number of layers the DGEB will use
    assert embeddings.shape == expected_shape


def test_ume_adapter_empty_sequences(test_model_checkpoint):
    """Test UMEAdapterDGEB with empty sequences."""
    adapter = UMEAdapterDGEB(
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
    """Test UMEAdapterDGEB with DNA modality."""
    adapter = UMEAdapterDGEB(
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
    expected_shape = (len(dna_sequences), 2, adapter.embed_dim)  # 2 is the number of layers the DGEB will use
    assert embeddings.shape == expected_shape


def test_ume_adapter_metadata(test_model_checkpoint):
    """Test UMEAdapterDGEB metadata property."""
    adapter = UMEAdapterDGEB(
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


def test_run_evaluation_task_metadata_extraction(monkeypatch):
    """Test that run_evaluation correctly extracts task_name and task_type from TaskMetadata objects in results."""

    # Create a fake TaskMetadata-like object
    class FakeTaskMetadata:
        display_name = "Fake Task"
        type = "fake_type"

    # Create a fake result object
    class FakeTaskMetric:
        id = "main_score"
        value = 0.99

    class FakeLayerResult:
        layer_number = 0
        metrics = [FakeTaskMetric()]

    class FakeResult:
        def __init__(self):
            self.task = FakeTaskMetadata()
            self.results = [FakeLayerResult()]

    # Create a fake UMEAdapterDGEB
    class FakeUMEAdapterDGEB:
        def __init__(self, **kwargs):
            self.embed_dim = 128
            self.num_layers = 6
            self.metadata = {"modality": "protein", "embed_dim": 128, "num_layers": 6}

    # Patch all the necessary components
    monkeypatch.setattr(
        dgeb_runner, "get_tasks_for_modality", lambda modality: [types.SimpleNamespace(metadata=FakeTaskMetadata)]
    )
    monkeypatch.setattr(dgeb_runner, "UMEAdapterDGEB", FakeUMEAdapterDGEB)

    class FakeDGEB:
        def __init__(self, tasks, seed):
            pass

        def run(self, model, output_folder=None):
            return [FakeResult()]

    monkeypatch.setattr(
        dgeb_runner,
        "dgeb",
        types.SimpleNamespace(
            DGEB=FakeDGEB, get_tasks_by_name=lambda names: [types.SimpleNamespace(metadata=FakeTaskMetadata)]
        ),
    )

    # Run evaluation
    results = dgeb_runner.run_evaluation(
        model_name="fake_model",
        modality="protein",
        tasks=None,
        output_dir="/tmp",
        batch_size=1,
        max_seq_length=10,
        use_flash_attn=False,
        l2_norm=False,
        pool_type="mean",
        devices=[0],
        seed=42,
    )
    assert results["results"][0]["task_name"] == "Fake Task"
    assert results["results"][0]["task_type"] == "fake_type"
    assert results["results"][0]["scores"]["layer_0"]["main_score"] == 0.99
