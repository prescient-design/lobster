"""Unit tests for utility tools in Lobster MCP server."""

import logging
from unittest.mock import Mock, patch

import pytest
import torch

from lobster.mcp.tools.tool_utils import _load_model, compute_naturalness, list_available_models


class TestListAvailableModels:
    """Test the list_available_models function."""

    def test_list_available_models_success(self):
        """Test successful listing of available models."""
        result = list_available_models()

        # Verify result structure
        assert isinstance(result, dict)
        assert "available_models" in result
        assert "device" in result
        assert "device_type" in result

        # Verify device is set
        assert result["device"] in ["cuda", "cpu", "mps"]
        assert result["device_type"] in ["cuda", "cpu", "mps"]

    def test_list_available_models_structure(self):
        """Test that the response has the expected structure."""
        result = list_available_models()

        # Check that available_models contains expected keys
        assert "masked_lm" in result["available_models"]
        assert "concept_bottleneck" in result["available_models"]

        # Check that each model type has models
        assert len(result["available_models"]["masked_lm"]) > 0
        assert len(result["available_models"]["concept_bottleneck"]) > 0


class TestComputeNaturalness:
    """Test the compute_naturalness function."""

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_with_naturalness_method(self, mock_load_model):
        """Test compute_naturalness with a model that has naturalness method."""
        # Mock model with naturalness method
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([0.8, 0.6])
        mock_load_model.return_value = mock_model

        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ]

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Verify result
        assert isinstance(result, dict)
        assert result["sequences"] == sequences
        # The function converts torch tensors to lists, so we expect the converted values
        # Use approximate equality for floating point values
        assert len(result["scores"]) == 2
        assert abs(result["scores"][0] - 0.8) < 1e-6
        assert abs(result["scores"][1] - 0.6) < 1e-6
        assert result["model_used"] == "masked_lm_test_model"

        # Verify model was called correctly
        mock_load_model.assert_called_once_with("test_model", "masked_lm")
        mock_model.naturalness.assert_called_once_with(sequences)

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_with_likelihood_method(self, mock_load_model):
        """Test compute_naturalness with a model that has likelihood method."""
        # Mock model with likelihood method but no naturalness
        mock_model = Mock()
        del mock_model.naturalness  # Remove naturalness method
        mock_model.likelihood.return_value = torch.tensor([0.7, 0.5])
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="concept_bottleneck")

        # Verify result
        assert isinstance(result, dict)
        assert result["sequences"] == sequences
        # Use approximate equality for floating point values
        assert len(result["scores"]) == 2
        assert abs(result["scores"][0] - 0.7) < 1e-6
        assert abs(result["scores"][1] - 0.5) < 1e-6
        assert result["model_used"] == "concept_bottleneck_test_model"

        # Verify model was called correctly
        mock_load_model.assert_called_once_with("test_model", "concept_bottleneck")
        mock_model.likelihood.assert_called_once_with(sequences)

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_with_list_scores(self, mock_load_model):
        """Test compute_naturalness with scores that are already a list."""
        # Mock model returning list instead of tensor
        mock_model = Mock()
        mock_model.naturalness.return_value = [0.9, 0.3]
        mock_load_model.return_value = mock_model

        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ]

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Verify result
        assert isinstance(result, dict)
        assert result["scores"] == [0.9, 0.3]

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_model_loading_error(self, mock_load_model):
        """Test compute_naturalness when model loading fails."""
        mock_load_model.side_effect = ValueError("Model not found")

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with pytest.raises(ValueError, match="Model not found"):
            compute_naturalness(model_name="invalid_model", sequences=sequences, model_type="masked_lm")

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_no_naturalness_or_likelihood_method(self, mock_load_model):
        """Test compute_naturalness with a model that has neither naturalness nor likelihood."""
        # Mock model without naturalness or likelihood methods
        mock_model = Mock()
        del mock_model.naturalness
        del mock_model.likelihood
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with pytest.raises(ValueError, match="does not support naturalness/likelihood computation"):
            compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_computation_error(self, mock_load_model):
        """Test compute_naturalness when computation fails."""
        # Mock model that raises an error during computation
        mock_model = Mock()
        mock_model.naturalness.side_effect = RuntimeError("Computation failed")
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with pytest.raises(RuntimeError, match="Computation failed"):
            compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_empty_sequences(self, mock_load_model):
        """Test compute_naturalness with empty sequences list."""
        # Mock model
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([])
        mock_load_model.return_value = mock_model

        sequences = []

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Verify result
        assert isinstance(result, dict)
        assert result["sequences"] == []
        assert result["scores"] == []
        assert result["model_used"] == "masked_lm_test_model"

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_single_sequence(self, mock_load_model):
        """Test compute_naturalness with a single sequence."""
        # Mock model
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([0.85])
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Verify result
        assert isinstance(result, dict)
        assert result["sequences"] == sequences
        # Use approximate equality for floating point values
        assert len(result["scores"]) == 1
        assert abs(result["scores"][0] - 0.85) < 1e-6
        assert result["model_used"] == "masked_lm_test_model"

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_logging_on_error(self, mock_load_model, caplog):
        """Test that errors are logged properly."""
        # Mock model that raises an error
        mock_model = Mock()
        mock_model.naturalness.side_effect = RuntimeError("Test error")
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with pytest.raises(RuntimeError):
            compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Check that error was logged
        assert "Error computing naturalness: Test error" in caplog.text

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_different_model_types(self, mock_load_model):
        """Test compute_naturalness with different model types."""
        # Mock model
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([0.8])
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Test masked_lm
        result1 = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")
        assert result1["model_used"] == "masked_lm_test_model"

        # Test concept_bottleneck
        result2 = compute_naturalness(model_name="test_model", sequences=sequences, model_type="concept_bottleneck")
        assert result2["model_used"] == "concept_bottleneck_test_model"


class TestLoadModel:
    """Test the _load_model utility function."""

    @patch("lobster.mcp.tools.tool_utils.AVAILABLE_MODELS")
    @patch("lobster.mcp.tools.tool_utils.LobsterPMLM")
    @patch("lobster.mcp.tools.tool_utils._get_device")
    def test_load_model_masked_lm(self, mock_get_device, mock_lobster_pmlm, mock_available_models):
        """Test loading a masked LM model."""
        mock_get_device.return_value = "cuda"
        # Set up the mock to return the expected structure
        mock_available_models.__getitem__.side_effect = (
            lambda key: {"test_model": "test/path"} if key == "masked_lm" else {}
        )
        mock_model_instance = Mock()
        # Ensure the mock model instance returns itself for chained calls
        mock_model_instance.to.return_value = mock_model_instance
        mock_lobster_pmlm.return_value = mock_model_instance

        result = _load_model("test_model", "masked_lm")

        # Verify model was created correctly
        mock_lobster_pmlm.assert_called_once_with("test/path")
        mock_model_instance.to.assert_called_once_with("cuda")
        mock_model_instance.eval.assert_called_once()
        assert result == mock_model_instance

    @patch("lobster.mcp.tools.tool_utils.AVAILABLE_MODELS")
    @patch("lobster.mcp.tools.tool_utils.LobsterCBMPMLM")
    @patch("lobster.mcp.tools.tool_utils._get_device")
    def test_load_model_concept_bottleneck(self, mock_get_device, mock_lobster_cbmpmlm, mock_available_models):
        """Test loading a concept bottleneck model."""
        mock_get_device.return_value = "cpu"
        mock_available_models.__getitem__.side_effect = (
            lambda key: {"test_model": "test/path"} if key == "concept_bottleneck" else {}
        )
        mock_model_instance = Mock()
        # Ensure the mock model instance returns itself for chained calls
        mock_model_instance.to.return_value = mock_model_instance
        mock_lobster_cbmpmlm.return_value = mock_model_instance

        result = _load_model("test_model", "concept_bottleneck")

        # Verify model was created correctly
        mock_lobster_cbmpmlm.assert_called_once_with("test/path")
        mock_model_instance.to.assert_called_once_with("cpu")
        mock_model_instance.eval.assert_called_once()
        assert result == mock_model_instance

    @patch("lobster.mcp.tools.tool_utils.AVAILABLE_MODELS")
    def test_load_model_unknown_model_name(self, mock_available_models):
        """Test loading a model with unknown name."""
        mock_available_models.__getitem__.return_value = {}

        with pytest.raises(ValueError, match="Unknown masked LM model: unknown_model"):
            _load_model("unknown_model", "masked_lm")

    def test_load_model_unknown_model_type(self):
        """Test loading a model with unknown type."""
        with pytest.raises(ValueError, match="Unknown model type: invalid_type"):
            _load_model("test_model", "invalid_type")


class TestToolUtilsIntegration:
    """Integration tests for tool utilities."""

    def test_list_available_models_workflow(self):
        """Test the complete workflow of listing available models."""
        result = list_available_models()

        # Verify basic structure
        assert isinstance(result, dict)
        assert "available_models" in result
        assert "device" in result
        assert "device_type" in result

        # Verify content
        assert isinstance(result["available_models"], dict)
        assert len(result["available_models"]) > 0

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_workflow(self, mock_load_model):
        """Test the complete workflow of computing naturalness."""
        # Mock model
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([0.8, 0.6])
        mock_load_model.return_value = mock_model

        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ]

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Verify complete workflow
        assert isinstance(result, dict)
        assert result["sequences"] == sequences
        # Use approximate equality for floating point values
        assert len(result["scores"]) == 2
        assert abs(result["scores"][0] - 0.8) < 1e-6
        assert abs(result["scores"][1] - 0.6) < 1e-6
        assert result["model_used"] == "masked_lm_test_model"

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_with_realistic_data(self, mock_load_model):
        """Test compute_naturalness with realistic protein sequences."""
        # Mock model
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([0.95, 0.23, 0.78])
        mock_load_model.return_value = mock_model

        # Realistic protein sequences
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "MKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
        ]

        result = compute_naturalness(model_name="lobster_24M", sequences=sequences, model_type="masked_lm")

        # Verify result
        assert isinstance(result, dict)
        assert result["sequences"] == sequences
        # Use approximate equality for floating point values
        assert len(result["scores"]) == 3
        assert abs(result["scores"][0] - 0.95) < 1e-6
        assert abs(result["scores"][1] - 0.23) < 1e-6
        assert abs(result["scores"][2] - 0.78) < 1e-6
        assert result["model_used"] == "masked_lm_lobster_24M"


class TestToolUtilsErrorHandling:
    """Test error handling in tool utilities."""

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_memory_error(self, mock_load_model):
        """Test compute_naturalness when memory error occurs."""
        # Mock model that raises memory error
        mock_model = Mock()
        mock_model.naturalness.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with pytest.raises(torch.cuda.OutOfMemoryError):
            compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_tensor_conversion_error(self, mock_load_model):
        """Test compute_naturalness when tensor conversion fails."""
        # Mock model returning invalid tensor
        mock_model = Mock()
        mock_tensor = Mock()
        # Make the mock tensor behave like a torch tensor
        mock_tensor.tolist.side_effect = RuntimeError("Tensor conversion failed")
        # Patch torch.is_tensor to return True for our mock tensor
        with patch("lobster.mcp.tools.tool_utils.torch.is_tensor", return_value=True):
            mock_model.naturalness.return_value = mock_tensor
            mock_load_model.return_value = mock_model

            sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

            with pytest.raises(RuntimeError, match="Tensor conversion failed"):
                compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_very_long_sequence(self, mock_load_model):
        """Test compute_naturalness with very long sequences."""
        # Mock model
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([0.5])
        mock_load_model.return_value = mock_model

        # Very long sequence
        long_sequence = "M" * 10000
        sequences = [long_sequence]

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Should handle long sequences
        assert isinstance(result, dict)
        assert result["sequences"] == sequences
        # Use approximate equality for floating point values
        assert len(result["scores"]) == 1
        assert abs(result["scores"][0] - 0.5) < 1e-6

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_compute_naturalness_mixed_sequence_lengths(self, mock_load_model):
        """Test compute_naturalness with sequences of different lengths."""
        # Mock model
        mock_model = Mock()
        mock_model.naturalness.return_value = torch.tensor([0.8, 0.3, 0.9])
        mock_load_model.return_value = mock_model

        # Sequences of different lengths
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Long
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Long
            "M",  # Short
        ]

        result = compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Should handle mixed lengths
        assert isinstance(result, dict)
        assert result["sequences"] == sequences
        # Use approximate equality for floating point values
        assert len(result["scores"]) == 3
        assert abs(result["scores"][0] - 0.8) < 1e-6
        assert abs(result["scores"][1] - 0.3) < 1e-6
        assert abs(result["scores"][2] - 0.9) < 1e-6


class TestToolUtilsLogging:
    """Test logging behavior in tool utilities."""

    def test_logging_configured(self):
        """Test that logging is properly configured."""
        logger = logging.getLogger("lobster-fastmcp-server")
        assert logger.level <= logging.INFO

    @patch("lobster.mcp.tools.tool_utils._load_model")
    def test_error_logging_format(self, mock_load_model, caplog):
        """Test that error logging has the correct format."""
        # Mock model that raises an error
        mock_model = Mock()
        mock_model.naturalness.side_effect = ValueError("Test error message")
        mock_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with pytest.raises(ValueError):
            compute_naturalness(model_name="test_model", sequences=sequences, model_type="masked_lm")

        # Check log format
        assert "Error computing naturalness: Test error message" in caplog.text
        assert "lobster-fastmcp-server" in caplog.text
