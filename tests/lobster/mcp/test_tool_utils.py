"""Tests for utility tools in Lobster MCP server."""

import pytest
import torch
from unittest.mock import Mock, patch

try:
    from lobster.mcp.tools.tool_utils import list_available_models, compute_naturalness
    from lobster.mcp.schemas import NaturalnessRequest
    from lobster.mcp.models import ModelManager, AVAILABLE_MODELS

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestListAvailableModels:
    """Test the list_available_models function."""

    def test_list_available_models_success(self):
        """Test successful listing of available models."""
        # Setup mock model manager
        mock_model_manager = Mock(spec=ModelManager)

        # Mock device info
        mock_device_info = {
            "device": "cuda:0",
            "device_type": "cuda",
            "gpu_memory_available": "8GB",
            "cuda_version": "11.8",
        }
        mock_model_manager.get_device_info.return_value = mock_device_info

        # Call function
        result = list_available_models(mock_model_manager)

        # Verify model manager calls
        mock_model_manager.get_device_info.assert_called_once()

        # Verify result structure
        assert "available_models" in result
        assert "device" in result
        assert "device_type" in result
        assert "gpu_memory_available" in result
        assert "cuda_version" in result

        # Verify result values
        assert result["available_models"] == AVAILABLE_MODELS
        assert result["device"] == "cuda:0"
        assert result["device_type"] == "cuda"
        assert result["gpu_memory_available"] == "8GB"
        assert result["cuda_version"] == "11.8"

    def test_list_available_models_cpu_device(self):
        """Test listing models with CPU device."""
        mock_model_manager = Mock(spec=ModelManager)

        mock_device_info = {"device": "cpu", "device_type": "cpu", "cpu_cores": 8, "memory_available": "16GB"}
        mock_model_manager.get_device_info.return_value = mock_device_info

        result = list_available_models(mock_model_manager)

        assert result["device"] == "cpu"
        assert result["device_type"] == "cpu"
        assert result["cpu_cores"] == 8
        assert result["memory_available"] == "16GB"
        assert result["available_models"] == AVAILABLE_MODELS

    def test_list_available_models_mps_device(self):
        """Test listing models with MPS device (Apple Silicon)."""
        mock_model_manager = Mock(spec=ModelManager)

        mock_device_info = {"device": "mps:0", "device_type": "mps", "mps_available": True}
        mock_model_manager.get_device_info.return_value = mock_device_info

        result = list_available_models(mock_model_manager)

        assert result["device"] == "mps:0"
        assert result["device_type"] == "mps"
        assert result["mps_available"] is True
        assert result["available_models"] == AVAILABLE_MODELS

    def test_list_available_models_empty_device_info(self):
        """Test listing models with empty device info."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_device_info.return_value = {}

        result = list_available_models(mock_model_manager)

        assert result["available_models"] == AVAILABLE_MODELS
        assert len(result) == 1  # Only available_models should be present

    def test_list_available_models_device_info_error(self):
        """Test handling of device info errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_device_info.side_effect = Exception("Device info failed")

        with pytest.raises(Exception, match="Device info failed"):
            list_available_models(mock_model_manager)

    def test_list_available_models_structure(self):
        """Test that the result structure is correct."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_device_info = {"device": "cuda:0", "device_type": "cuda"}
        mock_model_manager.get_device_info.return_value = mock_device_info

        result = list_available_models(mock_model_manager)

        # Verify the result is a dictionary
        assert isinstance(result, dict)

        # Verify available_models is present and is a dictionary
        assert "available_models" in result
        assert isinstance(result["available_models"], dict)

        # Verify device info is merged correctly
        assert result["device"] == "cuda:0"
        assert result["device_type"] == "cuda"


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestComputeNaturalness:
    """Test the compute_naturalness function."""

    def test_compute_naturalness_naturalness_method_success(self):
        """Test successful naturalness computation using naturalness method."""
        # Setup mock model manager and model
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock naturalness scores
        mock_scores = torch.tensor([0.85, 0.12, 0.73])
        mock_model.naturalness.return_value = mock_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        # Create request
        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "ACDEFGHIJKLMNOPQRSTUVWXYZ",
            ],
        )

        # Call function
        result = compute_naturalness(request, mock_model_manager)

        # Verify model manager calls
        mock_model_manager.get_or_load_model.assert_called_once_with("test_model", "transformer")

        # Verify model calls
        mock_model.naturalness.assert_called_once_with(request.sequences)

        # Verify result structure
        assert "sequences" in result
        assert "scores" in result
        assert "model_used" in result

        # Verify result values
        assert result["sequences"] == request.sequences
        assert result["scores"] == mock_scores.tolist()
        assert result["model_used"] == "transformer_test_model"
        assert len(result["scores"]) == 3

    def test_compute_naturalness_likelihood_method_success(self):
        """Test successful naturalness computation using likelihood method (fallback)."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock likelihood scores (no naturalness method)
        mock_scores = torch.tensor([0.92, 0.08])
        mock_model.likelihood.return_value = mock_scores
        # Don't have naturalness method
        del mock_model.naturalness
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="cnn",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ],
        )

        result = compute_naturalness(request, mock_model_manager)

        # Verify likelihood method was called (fallback)
        mock_model.likelihood.assert_called_once_with(request.sequences)

        assert result["scores"] == mock_scores.tolist()
        assert result["model_used"] == "cnn_test_model"
        assert len(result["scores"]) == 2

    def test_compute_naturalness_list_scores(self):
        """Test naturalness computation when model returns list instead of tensor."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock list scores
        mock_scores = [0.85, 0.12, 0.73]
        mock_model.naturalness.return_value = mock_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "ACDEFGHIJKLMNOPQRSTUVWXYZ",
            ],
        )

        result = compute_naturalness(request, mock_model_manager)

        # Verify scores are returned as-is (not converted to list)
        assert result["scores"] == mock_scores
        assert isinstance(result["scores"], list)

    def test_compute_naturalness_single_sequence(self):
        """Test naturalness computation for a single sequence."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_scores = torch.tensor([0.85])
        mock_model.naturalness.return_value = mock_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )

        result = compute_naturalness(request, mock_model_manager)

        assert len(result["sequences"]) == 1
        assert len(result["scores"]) == 1
        assert abs(result["scores"][0] - 0.85) < 1e-6  # Allow for floating point precision

    def test_compute_naturalness_empty_sequences(self):
        """Test naturalness computation with empty sequence list."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_scores = torch.tensor([])
        mock_model.naturalness.return_value = mock_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(model_name="test_model", model_type="transformer", sequences=[])

        result = compute_naturalness(request, mock_model_manager)

        assert result["sequences"] == []
        assert result["scores"] == []
        assert len(result["scores"]) == 0

    def test_compute_naturalness_no_methods_available(self):
        """Test handling when neither naturalness nor likelihood methods are available."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Remove both methods
        del mock_model.naturalness
        del mock_model.likelihood
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )

        with pytest.raises(ValueError, match="Model test_model does not support naturalness/likelihood computation"):
            compute_naturalness(request, mock_model_manager)

    def test_compute_naturalness_model_loading_error(self):
        """Test handling of model loading errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Model not found")

        request = NaturalnessRequest(
            model_name="nonexistent_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )

        with pytest.raises(Exception, match="Model not found"):
            compute_naturalness(request, mock_model_manager)

    def test_compute_naturalness_computation_error(self):
        """Test handling of naturalness computation errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()
        mock_model.naturalness.side_effect = Exception("Computation failed")
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )

        with pytest.raises(Exception, match="Computation failed"):
            compute_naturalness(request, mock_model_manager)

    def test_compute_naturalness_error_logging(self):
        """Test that errors are properly logged."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Test error")

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )

        with patch("lobster.mcp.tools.tool_utils.logger") as mock_logger:
            with pytest.raises(Exception, match="Test error"):
                compute_naturalness(request, mock_model_manager)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Error computing naturalness" in mock_logger.error.call_args[0][0]

    def test_compute_naturalness_request_validation(self):
        """Test that request objects are properly validated."""
        # Test valid request with all parameters
        valid_request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ],
        )
        assert valid_request.model_name == "test_model"
        assert valid_request.model_type == "transformer"
        assert len(valid_request.sequences) == 2

    def test_compute_naturalness_different_model_types(self):
        """Test naturalness computation with different model types."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_scores = torch.tensor([0.85])
        mock_model.naturalness.return_value = mock_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        model_types = ["transformer", "cnn", "lstm", "bert", "gpt"]

        for model_type in model_types:
            request = NaturalnessRequest(
                model_name="test_model",
                model_type=model_type,
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            )

            result = compute_naturalness(request, mock_model_manager)

            assert result["model_used"] == f"{model_type}_test_model"
            assert len(result["scores"]) == 1

    def test_compute_naturalness_score_ranges(self):
        """Test naturalness computation with different score ranges."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Test with high scores (very natural sequences)
        high_scores = torch.tensor([0.95, 0.98, 0.99])
        mock_model.naturalness.return_value = high_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "ACDEFGHIJKLMNOPQRSTUVWXYZ",
                "QWERTYUIOPASDFGHJKLZXCVBNM",
            ],
        )

        result = compute_naturalness(request, mock_model_manager)

        assert len(result["scores"]) == 3
        assert all(abs(score - expected) < 1e-6 for score, expected in zip(result["scores"], [0.95, 0.98, 0.99]))
        assert all(0 <= score <= 1 for score in result["scores"])

        # Test with low scores (artificial sequences)
        low_scores = torch.tensor([0.01, 0.05, 0.1])
        mock_model.naturalness.return_value = low_scores

        result = compute_naturalness(request, mock_model_manager)

        assert len(result["scores"]) == 3
        assert all(abs(score - expected) < 1e-6 for score, expected in zip(result["scores"], [0.01, 0.05, 0.1]))
        assert all(0 <= score <= 1 for score in result["scores"])

    def test_compute_naturalness_mixed_score_types(self):
        """Test naturalness computation with mixed tensor and list returns."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Test with numpy array
        import numpy as np

        numpy_scores = np.array([0.85, 0.12])
        mock_model.naturalness.return_value = numpy_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ],
        )

        result = compute_naturalness(request, mock_model_manager)

        # Numpy arrays are not converted to lists by the function
        assert len(result["scores"]) == 2
        assert all(abs(score - expected) < 1e-6 for score, expected in zip(result["scores"], [0.85, 0.12]))
        # The function only converts PyTorch tensors, not numpy arrays
        assert not isinstance(result["scores"], list)  # Should remain numpy array


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestToolUtilsIntegration:
    """Integration tests for utility tools."""

    def test_utils_workflow(self):
        """Test a complete workflow: list models, then compute naturalness."""
        mock_model_manager = Mock(spec=ModelManager)

        # Setup for list_available_models
        mock_device_info = {"device": "cuda:0", "device_type": "cuda", "gpu_memory_available": "8GB"}
        mock_model_manager.get_device_info.return_value = mock_device_info

        # Setup for compute_naturalness
        mock_model = Mock()
        mock_scores = torch.tensor([0.85, 0.12])
        mock_model.naturalness.return_value = mock_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        # Step 1: List available models
        models_result = list_available_models(mock_model_manager)

        assert "available_models" in models_result
        assert models_result["device"] == "cuda:0"

        # Step 2: Compute naturalness
        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ],
        )

        naturalness_result = compute_naturalness(request, mock_model_manager)

        assert len(naturalness_result["scores"]) == 2
        assert all(abs(score - expected) < 1e-6 for score, expected in zip(naturalness_result["scores"], [0.85, 0.12]))
        assert len(naturalness_result["sequences"]) == 2

        # Verify the same model manager was used for both calls
        assert mock_model_manager.get_device_info.call_count == 1
        assert mock_model_manager.get_or_load_model.call_count == 1

    def test_utils_error_handling_consistency(self):
        """Test that both functions handle errors consistently."""
        mock_model_manager = Mock(spec=ModelManager)

        # Test list_available_models error
        mock_model_manager.get_device_info.side_effect = Exception("Device error")

        with pytest.raises(Exception, match="Device error"):
            list_available_models(mock_model_manager)

        # Reset and test compute_naturalness error
        mock_model_manager.reset_mock()
        mock_model_manager.get_or_load_model.side_effect = Exception("Model error")

        request = NaturalnessRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )

        with pytest.raises(Exception, match="Model error"):
            compute_naturalness(request, mock_model_manager)
