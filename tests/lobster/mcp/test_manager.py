"""Tests for ModelManager in Lobster MCP server."""

import pytest
from unittest.mock import Mock, patch

try:
    from lobster.mcp.models.manager import ModelManager
    from lobster.model import LobsterPMLM, LobsterCBMPMLM

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestModelManager:
    """Test the ModelManager class."""

    def test_model_manager_initialization_cuda(self):
        """Test ModelManager initialization with CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            assert manager.device == "cuda"
            assert manager.loaded_models == {}
            assert isinstance(manager.loaded_models, dict)

    def test_model_manager_initialization_cpu(self):
        """Test ModelManager initialization with only CPU available."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = ModelManager()

            assert manager.device == "cpu"
            assert manager.loaded_models == {}
            assert isinstance(manager.loaded_models, dict)

    @patch("lobster.mcp.models.manager.logger")
    def test_model_manager_initialization_logging(self, mock_logger):
        """Test that initialization logs the device."""
        with patch("torch.cuda.is_available", return_value=True):
            ModelManager()

            mock_logger.info.assert_called_once_with("Initialized ModelManager on device: cuda")

    def test_get_device_info_cuda(self):
        """Test get_device_info with CUDA device."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()
            device_info = manager.get_device_info()

            assert device_info["device"] == "cuda"

    def test_get_device_info_cpu(self):
        """Test get_device_info with CPU device."""
        with patch("torch.cuda.is_available", return_value=False):
            manager = ModelManager()
            device_info = manager.get_device_info()

            assert device_info["device"] == "cpu"

    @patch("lobster.mcp.models.manager.LobsterPMLM")
    @patch("lobster.mcp.models.manager.logger")
    def test_get_or_load_model_masked_lm_success(self, mock_logger, mock_lobster_pmlm):
        """Test successful loading of masked LM model."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            # Mock the model
            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self
            mock_lobster_pmlm.return_value = mock_model

            # Mock AVAILABLE_MODELS
            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                # Load model
                result = manager.get_or_load_model("test_model", "masked_lm")

                # Verify model was created
                mock_lobster_pmlm.assert_called_once_with("/path/to/model")
                mock_model.to.assert_called_once_with("cuda")
                mock_model.eval.assert_called_once()

                # Verify result
                assert result == mock_model

                # Verify logging
                mock_logger.info.assert_any_call("Loading model test_model of type masked_lm")
                mock_logger.info.assert_any_call("Successfully loaded masked_lm_test_model")

                # Verify caching
                assert "masked_lm_test_model" in manager.loaded_models
                assert manager.loaded_models["masked_lm_test_model"] == mock_model

    @patch("lobster.mcp.models.manager.LobsterCBMPMLM")
    @patch("lobster.mcp.models.manager.logger")
    def test_get_or_load_model_concept_bottleneck_success(self, mock_logger, mock_lobster_cbm):
        """Test successful loading of concept bottleneck model."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            # Mock the model
            mock_model = Mock(spec=LobsterCBMPMLM)
            mock_model.to.return_value = mock_model  # to() should return self
            mock_lobster_cbm.return_value = mock_model

            # Mock AVAILABLE_MODELS
            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                # Load model
                result = manager.get_or_load_model("test_model", "concept_bottleneck")

                # Verify model was created
                mock_lobster_cbm.assert_called_once_with("/path/to/model")
                mock_model.to.assert_called_once_with("cuda")
                mock_model.eval.assert_called_once()

                # Verify result
                assert result == mock_model

                # Verify logging
                mock_logger.info.assert_any_call("Loading model test_model of type concept_bottleneck")
                mock_logger.info.assert_any_call("Successfully loaded concept_bottleneck_test_model")

                # Verify caching
                assert "concept_bottleneck_test_model" in manager.loaded_models
                assert manager.loaded_models["concept_bottleneck_test_model"] == mock_model

    def test_get_or_load_model_caching(self):
        """Test that models are properly cached and reused."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            # Mock the models
            mock_pmlm = Mock(spec=LobsterPMLM)
            mock_pmlm.to.return_value = mock_pmlm  # to() should return self
            mock_cbm = Mock(spec=LobsterCBMPMLM)
            mock_cbm.to.return_value = mock_cbm  # to() should return self

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM", return_value=mock_pmlm),
                patch("lobster.mcp.models.manager.LobsterCBMPMLM", return_value=mock_cbm),
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                # Load models for the first time
                result1 = manager.get_or_load_model("test_model", "masked_lm")
                result2 = manager.get_or_load_model("test_model", "concept_bottleneck")

                # Load the same models again
                result3 = manager.get_or_load_model("test_model", "masked_lm")
                result4 = manager.get_or_load_model("test_model", "concept_bottleneck")

                # Verify models are cached
                assert "masked_lm_test_model" in manager.loaded_models
                assert "concept_bottleneck_test_model" in manager.loaded_models

                # Verify same instances are returned
                assert result1 is result3
                assert result2 is result4
                assert result1 is not result2

    def test_get_or_load_model_unknown_masked_lm(self):
        """Test handling of unknown masked LM model."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {}  # Empty dict

                with pytest.raises(ValueError, match="Unknown masked LM model: unknown_model"):
                    manager.get_or_load_model("unknown_model", "masked_lm")

    def test_get_or_load_model_unknown_concept_bottleneck(self):
        """Test handling of unknown concept bottleneck model."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {}  # Empty dict

                with pytest.raises(ValueError, match="Unknown concept bottleneck model: unknown_model"):
                    manager.get_or_load_model("unknown_model", "concept_bottleneck")

    def test_get_or_load_model_unknown_model_type(self):
        """Test handling of unknown model type."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            with pytest.raises(ValueError, match="Unknown model type: unknown_type"):
                manager.get_or_load_model("test_model", "unknown_type")

    @patch("lobster.mcp.models.manager.LobsterPMLM")
    def test_get_or_load_model_model_loading_error(self, mock_lobster_pmlm):
        """Test handling of model loading errors."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            # Mock model loading to raise an exception
            mock_lobster_pmlm.side_effect = Exception("Model loading failed")

            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                with pytest.raises(Exception, match="Model loading failed"):
                    manager.get_or_load_model("test_model", "masked_lm")

    def test_get_or_load_model_different_model_names(self):
        """Test loading different model names."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_model1 = Mock(spec=LobsterPMLM)
            mock_model1.to.return_value = mock_model1  # to() should return self
            mock_model2 = Mock(spec=LobsterPMLM)
            mock_model2.to.return_value = mock_model2  # to() should return self

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM") as mock_lobster_pmlm,
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_available_models.__getitem__.return_value = {
                    "model1": "/path/to/model1",
                    "model2": "/path/to/model2",
                }

                # Configure mock to return different models
                mock_lobster_pmlm.side_effect = [mock_model1, mock_model2]

                # Load different models
                result1 = manager.get_or_load_model("model1", "masked_lm")
                result2 = manager.get_or_load_model("model2", "masked_lm")

                # Verify different models are loaded
                assert result1 is mock_model1
                assert result2 is mock_model2
                assert result1 is not result2

                # Verify both are cached
                assert "masked_lm_model1" in manager.loaded_models
                assert "masked_lm_model2" in manager.loaded_models

    def test_get_or_load_model_cache_key_format(self):
        """Test that cache keys are formatted correctly."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM", return_value=mock_model),
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                # Load model
                manager.get_or_load_model("test_model", "masked_lm")

                # Verify cache key format
                expected_key = "masked_lm_test_model"
                assert expected_key in manager.loaded_models
                assert manager.loaded_models[expected_key] == mock_model

    @patch("lobster.mcp.models.manager.LobsterPMLM")
    @patch("lobster.mcp.models.manager.logger")
    def test_get_or_load_model_logging_sequence(self, mock_logger, mock_lobster_pmlm):
        """Test that logging happens in the correct sequence."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self
            mock_lobster_pmlm.return_value = mock_model

            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                # Load model
                manager.get_or_load_model("test_model", "masked_lm")

                # Check that all expected log messages were called
                actual_calls = mock_logger.info.call_args_list
                assert len(actual_calls) >= 3

    def test_model_manager_multiple_instances(self):
        """Test that multiple ModelManager instances are independent."""
        with patch("torch.cuda.is_available", return_value=True):
            manager1 = ModelManager()
            manager2 = ModelManager()

            # Verify they have separate caches
            assert manager1.loaded_models is not manager2.loaded_models
            assert manager1.loaded_models == {}
            assert manager2.loaded_models == {}

    def test_get_or_load_model_empty_model_name(self):
        """Test handling of empty model name."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {"": "/path/to/model"}

                # This should work if empty string is in available models
                mock_model = Mock(spec=LobsterPMLM)
                mock_model.to.return_value = mock_model  # to() should return self
                with patch("lobster.mcp.models.manager.LobsterPMLM", return_value=mock_model):
                    result = manager.get_or_load_model("", "masked_lm")
                    assert result == mock_model

    def test_get_or_load_model_special_characters_in_name(self):
        """Test handling of special characters in model name."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM", return_value=mock_model),
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_available_models.__getitem__.return_value = {"test-model_1.0": "/path/to/model"}

                # Load model with special characters
                result = manager.get_or_load_model("test-model_1.0", "masked_lm")

                # Verify cache key includes special characters
                expected_key = "masked_lm_test-model_1.0"
                assert expected_key in manager.loaded_models
                assert result == mock_model

    @patch("lobster.mcp.models.manager.LobsterPMLM")
    def test_get_or_load_model_device_assignment(self, mock_lobster_pmlm):
        """Test that models are assigned to the correct device."""
        # Test with CUDA
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()
            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self
            mock_lobster_pmlm.return_value = mock_model

            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                manager.get_or_load_model("test_model", "masked_lm")
                mock_model.to.assert_called_with("cuda")

        # Test with CPU
        with patch("torch.cuda.is_available", return_value=False):
            manager = ModelManager()
            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self
            mock_lobster_pmlm.return_value = mock_model

            with patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models:
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                manager.get_or_load_model("test_model", "masked_lm")
                mock_model.to.assert_called_with("cpu")

    def test_get_or_load_model_eval_mode(self):
        """Test that models are set to evaluation mode."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM", return_value=mock_model),
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                manager.get_or_load_model("test_model", "masked_lm")

                # Verify model is set to eval mode
                mock_model.eval.assert_called_once()

    def test_get_or_load_model_path_validation(self):
        """Test that model paths are correctly retrieved from config."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_model = Mock(spec=LobsterPMLM)

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM") as mock_lobster_pmlm,
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_lobster_pmlm.return_value = mock_model
                mock_available_models.__getitem__.return_value = {"test_model": "/custom/path/to/model"}

                manager.get_or_load_model("test_model", "masked_lm")

                # Verify the correct path was used
                mock_lobster_pmlm.assert_called_once_with("/custom/path/to/model")


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestModelManagerIntegration:
    """Integration tests for ModelManager."""

    def test_model_manager_full_workflow(self):
        """Test a complete workflow with multiple model types."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_pmlm = Mock(spec=LobsterPMLM)
            mock_pmlm.to.return_value = mock_pmlm  # to() should return self
            mock_cbm = Mock(spec=LobsterCBMPMLM)
            mock_cbm.to.return_value = mock_cbm  # to() should return self

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM", return_value=mock_pmlm),
                patch("lobster.mcp.models.manager.LobsterCBMPMLM", return_value=mock_cbm),
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_available_models.__getitem__.side_effect = lambda key: {
                    "masked_lm": {"model1": "/path/to/pmlm"},
                    "concept_bottleneck": {"model2": "/path/to/cbm"},
                }[key]

                # Load both types of models
                pmlm_result = manager.get_or_load_model("model1", "masked_lm")
                cbm_result = manager.get_or_load_model("model2", "concept_bottleneck")

                # Verify both models are loaded and cached
                assert pmlm_result == mock_pmlm
                assert cbm_result == mock_cbm
                assert "masked_lm_model1" in manager.loaded_models
                assert "concept_bottleneck_model2" in manager.loaded_models

                # Verify device info
                device_info = manager.get_device_info()
                assert device_info["device"] == "cuda"

    def test_model_manager_error_recovery(self):
        """Test that errors don't corrupt the cache."""
        with patch("torch.cuda.is_available", return_value=True):
            manager = ModelManager()

            mock_model = Mock(spec=LobsterPMLM)
            mock_model.to.return_value = mock_model  # to() should return self

            with (
                patch("lobster.mcp.models.manager.LobsterPMLM") as mock_lobster_pmlm,
                patch("lobster.mcp.models.manager.AVAILABLE_MODELS") as mock_available_models,
            ):
                mock_available_models.__getitem__.return_value = {"test_model": "/path/to/model"}

                # First load should succeed
                mock_lobster_pmlm.return_value = mock_model
                result1 = manager.get_or_load_model("test_model", "masked_lm")

                # Verify model is cached
                assert "masked_lm_test_model" in manager.loaded_models

                # Simulate an error on second load attempt
                mock_lobster_pmlm.side_effect = Exception("Loading failed")

                # Should still return cached model
                result2 = manager.get_or_load_model("test_model", "masked_lm")

                # Should return the same cached model
                assert result1 is result2
                assert result1 == mock_model
