"""Tests for the DGEB evaluation callback - focused on callback-specific functionality."""

from unittest.mock import Mock, patch
import tempfile

import lightning as L
import pytest

from lobster.callbacks import DGEBEvaluationCallback


class MockModule(L.LightningModule):
    """Simple mock module for testing."""

    def state_dict(self):
        return {"mock": "data"}


@pytest.fixture
def mock_trainer():
    """Create a mock trainer."""
    trainer = Mock(spec=L.Trainer)
    trainer.save_checkpoint = Mock()
    return trainer


class TestDGEBEvaluationCallback:
    """Test suite for DGEBEvaluationCallback - core functionality only."""

    def test_initialization(self):
        """Test callback initialization with defaults and custom parameters."""
        # Test defaults
        callback = DGEBEvaluationCallback()
        assert callback.model_name == "UME"
        assert callback.modality == "protein"
        assert callback.requires_tokenization is True

        # Test custom parameters
        custom_callback = DGEBEvaluationCallback(model_name="custom-model", modality="dna", requires_tokenization=False)
        assert custom_callback.model_name == "custom-model"
        assert custom_callback.modality == "dna"
        assert custom_callback.requires_tokenization is False

    def test_routing_logic(self):
        """Test that callback routes UME vs ESM models correctly."""
        module = MockModule()

        # UME callback should use checkpoint-based evaluation
        ume_callback = DGEBEvaluationCallback(requires_tokenization=True)
        with patch.object(ume_callback, "_evaluate_esm_direct") as mock_esm_eval:
            with patch("torch.save"), patch("lobster.callbacks._dgeb_evaluation_callback.run_evaluation"):
                try:
                    ume_callback.evaluate(module, None)
                except Exception:
                    pass  # Expected to fail due to mocking
            # Should NOT call ESM evaluation
            mock_esm_eval.assert_not_called()

        # ESM callback should use direct evaluation
        esm_callback = DGEBEvaluationCallback(requires_tokenization=False)
        with patch.object(esm_callback, "_evaluate_esm_direct", return_value={}) as mock_esm_eval:
            esm_callback.evaluate(module, None)
            # Should call ESM evaluation
            mock_esm_eval.assert_called_once()

    @patch("lobster.callbacks._dgeb_evaluation_callback.run_evaluation")
    @patch("lobster.callbacks._dgeb_evaluation_callback.generate_report")
    @patch("torch.save")
    def test_ume_evaluation_flow(self, mock_torch_save, mock_generate_report, mock_run_evaluation, mock_trainer):
        """Test UME model evaluation flow."""
        callback = DGEBEvaluationCallback(model_name="test-ume")
        module = MockModule()

        mock_results = {"model_name": "test-ume", "results": []}
        mock_run_evaluation.return_value = mock_results

        with tempfile.TemporaryDirectory() as temp_dir:
            callback.output_dir = temp_dir
            results = callback.evaluate(module, mock_trainer)

        # Verify checkpoint-based evaluation was used
        mock_trainer.save_checkpoint.assert_called_once()
        mock_run_evaluation.assert_called_once()
        assert "_metadata" in results

    def test_esm_evaluation_flow(self):
        """Test ESM model evaluation flow."""
        callback = DGEBEvaluationCallback(model_name="test-esm", requires_tokenization=False)
        module = MockModule()

        expected_results = {"_metadata": {"model_name": "test-esm"}}

        with patch.object(callback, "_evaluate_esm_direct", return_value=expected_results) as mock_eval:
            results = callback.evaluate(module, None)

            # Verify direct evaluation was used
            mock_eval.assert_called_once_with(module, None)
            assert results == expected_results

    def test_error_handling(self):
        """Test that errors are handled gracefully."""
        callback = DGEBEvaluationCallback(requires_tokenization=False)
        module = MockModule()

        error_results = {"error": "DGEB library not available", "_metadata": {"model_name": callback.model_name}}

        with patch.object(callback, "_evaluate_esm_direct", return_value=error_results):
            results = callback.evaluate(module, None)

        assert "error" in results
        assert "_metadata" in results
