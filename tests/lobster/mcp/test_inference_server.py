"""
Tests for the Lobster MCP inference server
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directories to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from lobster.mcp.models import AVAILABLE_MODELS, ModelManager

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestModelManager:
    """Test the Lobster MCP ModelManager"""

    def test_initialization(self):
        """Test ModelManager initialization"""
        model_manager = ModelManager()
        assert hasattr(model_manager, "device")
        assert hasattr(model_manager, "loaded_models")
        assert isinstance(model_manager.loaded_models, dict)

    def test_available_models(self):
        """Test that available models are properly defined"""
        assert isinstance(AVAILABLE_MODELS, dict)
        assert "masked_lm" in AVAILABLE_MODELS
        assert "concept_bottleneck" in AVAILABLE_MODELS

        # Check specific models
        assert "lobster_24M" in AVAILABLE_MODELS["masked_lm"]
        assert "cb_lobster_24M" in AVAILABLE_MODELS["concept_bottleneck"]

    @patch("lobster.mcp.models.manager.LobsterPMLM")
    @patch("lobster.mcp.models.manager.torch")
    def test_model_loading_masked_lm(self, mock_torch, mock_lobster_pmlm):
        """Test loading masked language model"""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_lobster_pmlm.return_value = mock_model
        mock_model.to.return_value = mock_model

        model_manager = ModelManager()

        # Test loading
        model = model_manager.get_or_load_model("lobster_24M", "masked_lm")

        # Verify calls
        mock_lobster_pmlm.assert_called_once_with("asalam91/lobster_24M")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

        assert model == mock_model

    @patch("lobster.mcp.models.manager.LobsterCBMPMLM")
    @patch("lobster.mcp.models.manager.torch")
    def test_model_loading_concept_bottleneck(self, mock_torch, mock_lobster_cbm):
        """Test loading concept bottleneck model"""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_lobster_cbm.return_value = mock_model
        mock_model.to.return_value = mock_model

        model_manager = ModelManager()

        # Test loading
        model = model_manager.get_or_load_model("cb_lobster_24M", "concept_bottleneck")

        # Verify calls
        mock_lobster_cbm.assert_called_once_with("asalam91/cb_lobster_24M")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

        assert model == mock_model

    def test_invalid_model_type(self):
        """Test error handling for invalid model type"""
        model_manager = ModelManager()

        with pytest.raises(ValueError, match="Unknown model type"):
            model_manager.get_or_load_model("some_model", "invalid_type")

    def test_invalid_model_name(self):
        """Test error handling for invalid model name"""
        model_manager = ModelManager()

        with pytest.raises(ValueError, match="Unknown masked LM model"):
            model_manager.get_or_load_model("invalid_model", "masked_lm")


class TestModuleImports:
    """Test module imports and availability"""

    def test_torch_import(self):
        """Test that torch can be imported"""
        try:
            import torch

            assert hasattr(torch, "__version__")
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_mcp_import(self):
        """Test that MCP can be imported"""
        try:
            from mcp import Server
            from mcp.types import Tool

            # Basic smoke test - if we can import these, MCP is working
            assert Server is not None
            assert Tool is not None
        except ImportError:
            pytest.skip("MCP not available")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestIntegration:
    """Integration tests that require actual model loading"""

    @pytest.fixture
    def model_manager(self):
        """Create a ModelManager instance for testing"""
        return ModelManager()

    @pytest.mark.gpu
    def test_device_detection(self, model_manager):
        """Test device detection"""
        assert model_manager.device in ["cuda", "cpu"]

    # Note: Actual model loading tests would require significant resources
    # and internet connectivity, so they're marked as slow/integration tests
