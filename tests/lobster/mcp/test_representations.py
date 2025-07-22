"""Tests for representation-related tools in Lobster MCP server."""

import pytest
import torch
from unittest.mock import Mock, patch

try:
    from lobster.mcp.tools.representations import get_sequence_representations
    from lobster.mcp.schemas import SequenceRepresentationRequest
    from lobster.mcp.models import ModelManager

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestGetSequenceRepresentations:
    """Test the get_sequence_representations function."""

    def test_get_sequence_representations_cls_success(self):
        """Test successful CLS token representation extraction."""
        # Setup mock model manager and model
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock representations tensor (batch_size=2, seq_len=10, hidden_dim=8)
        mock_representations = torch.randn(2, 10, 8)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        # Create request
        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "ACDEFGHIJKLMNOPQRSTUVWXYZ",
            ],
            representation_type="cls",
        )

        # Call function
        result = get_sequence_representations(request, mock_model_manager)

        # Verify model manager calls
        mock_model_manager.get_or_load_model.assert_called_once_with("test_model", "transformer")

        # Verify model calls
        mock_model.sequences_to_latents.assert_called_once_with(request.sequences)

        # Verify result structure
        assert "embeddings" in result
        assert "embedding_dimension" in result
        assert "num_sequences" in result
        assert "representation_type" in result
        assert "model_used" in result

        # Verify result values
        assert result["num_sequences"] == 2
        assert result["embedding_dimension"] == 8
        assert result["representation_type"] == "cls"
        assert result["model_used"] == "transformer_test_model"

        # Verify embeddings are CLS token representations (first token)
        expected_cls_embeddings = mock_representations[:, 0, :].cpu().numpy()
        assert result["embeddings"] == expected_cls_embeddings.tolist()
        assert len(result["embeddings"]) == 2
        assert len(result["embeddings"][0]) == 8

    def test_get_sequence_representations_pooled_success(self):
        """Test successful pooled representation extraction."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock representations tensor
        mock_representations = torch.randn(1, 15, 12)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="cnn",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="pooled",
        )

        result = get_sequence_representations(request, mock_model_manager)

        # Verify pooled representation (mean across sequence dimension)
        expected_pooled_embeddings = torch.mean(mock_representations, dim=1).cpu().numpy()
        assert result["embeddings"] == expected_pooled_embeddings.tolist()
        assert result["representation_type"] == "pooled"
        assert result["embedding_dimension"] == 12
        assert result["num_sequences"] == 1

    def test_get_sequence_representations_full_success(self):
        """Test successful full sequence representation extraction."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock representations tensor
        mock_representations = torch.randn(1, 20, 16)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="lstm",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="full",
        )

        result = get_sequence_representations(request, mock_model_manager)

        # Verify full sequence representation
        expected_full_embeddings = mock_representations.cpu().numpy()
        assert result["embeddings"] == expected_full_embeddings.tolist()
        assert result["representation_type"] == "full"
        assert result["embedding_dimension"] == 16
        assert result["num_sequences"] == 1

        # Verify shape: [num_sequences, seq_len, hidden_dim]
        assert len(result["embeddings"]) == 1  # num_sequences
        assert len(result["embeddings"][0]) == 20  # seq_len
        assert len(result["embeddings"][0][0]) == 16  # hidden_dim

    def test_get_sequence_representations_single_sequence(self):
        """Test representation extraction for a single sequence."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_representations = torch.randn(1, 10, 6)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )

        result = get_sequence_representations(request, mock_model_manager)

        assert result["num_sequences"] == 1
        assert len(result["embeddings"]) == 1
        assert result["embedding_dimension"] == 6

    def test_get_sequence_representations_empty_sequences(self):
        """Test representation extraction with empty sequence list."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Empty tensor for empty sequences
        mock_representations = torch.empty(0, 10, 8)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model", model_type="transformer", sequences=[], representation_type="cls"
        )

        result = get_sequence_representations(request, mock_model_manager)

        assert result["num_sequences"] == 0
        assert result["embeddings"] == []
        assert result["embedding_dimension"] == 8

    def test_get_sequence_representations_unknown_type(self):
        """Test handling of unknown representation type."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_representations = torch.randn(1, 10, 8)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="unknown_type",
        )

        with pytest.raises(ValueError, match="Unknown representation type: unknown_type"):
            get_sequence_representations(request, mock_model_manager)

    def test_get_sequence_representations_model_loading_error(self):
        """Test handling of model loading errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Model not found")

        request = SequenceRepresentationRequest(
            model_name="nonexistent_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )

        with pytest.raises(Exception, match="Model not found"):
            get_sequence_representations(request, mock_model_manager)

    def test_get_sequence_representations_model_inference_error(self):
        """Test handling of model inference errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()
        mock_model.sequences_to_latents.side_effect = Exception("Inference failed")
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )

        with pytest.raises(Exception, match="Inference failed"):
            get_sequence_representations(request, mock_model_manager)

    def test_get_sequence_representations_torch_no_grad_context(self):
        """Test that torch.no_grad() is used during inference."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_representations = torch.randn(1, 10, 8)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )

        # Mock torch.no_grad to verify it's called
        with patch("torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()

            get_sequence_representations(request, mock_model_manager)

            # Verify torch.no_grad was called
            mock_no_grad.assert_called_once()

    def test_get_sequence_representations_error_logging(self):
        """Test that errors are properly logged."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Test error")

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )

        with patch("lobster.mcp.tools.representations.logger") as mock_logger:
            with pytest.raises(Exception, match="Test error"):
                get_sequence_representations(request, mock_model_manager)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Error getting representations" in mock_logger.error.call_args[0][0]

    def test_get_sequence_representations_request_validation(self):
        """Test that request objects are properly validated."""
        # Test valid request with all parameters
        valid_request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )
        assert valid_request.model_name == "test_model"
        assert valid_request.model_type == "transformer"
        assert len(valid_request.sequences) == 1
        assert valid_request.representation_type == "cls"

        # Test valid request with default representation_type
        valid_request_default = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )
        assert valid_request_default.representation_type == "pooled"  # default

    def test_get_sequence_representations_different_model_types(self):
        """Test representation extraction with different model types."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_representations = torch.randn(1, 10, 8)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        model_types = ["transformer", "cnn", "lstm", "bert", "gpt"]

        for model_type in model_types:
            request = SequenceRepresentationRequest(
                model_name="test_model",
                model_type=model_type,
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                representation_type="cls",
            )

            result = get_sequence_representations(request, mock_model_manager)

            assert result["model_used"] == f"{model_type}_test_model"
            assert result["num_sequences"] == 1
            assert result["embedding_dimension"] == 8

    def test_get_sequence_representations_different_representation_types(self):
        """Test all three representation types with the same model."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Use consistent tensor for all tests
        mock_representations = torch.randn(1, 12, 10)
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Test CLS representation
        request_cls = SequenceRepresentationRequest(
            model_name="test_model", model_type="transformer", sequences=sequences, representation_type="cls"
        )

        result_cls = get_sequence_representations(request_cls, mock_model_manager)
        expected_cls = mock_representations[:, 0, :].cpu().numpy()
        assert result_cls["embeddings"] == expected_cls.tolist()
        assert result_cls["representation_type"] == "cls"

        # Test pooled representation
        request_pooled = SequenceRepresentationRequest(
            model_name="test_model", model_type="transformer", sequences=sequences, representation_type="pooled"
        )

        result_pooled = get_sequence_representations(request_pooled, mock_model_manager)
        expected_pooled = torch.mean(mock_representations, dim=1).cpu().numpy()
        assert result_pooled["embeddings"] == expected_pooled.tolist()
        assert result_pooled["representation_type"] == "pooled"

        # Test full representation
        request_full = SequenceRepresentationRequest(
            model_name="test_model", model_type="transformer", sequences=sequences, representation_type="full"
        )

        result_full = get_sequence_representations(request_full, mock_model_manager)
        expected_full = mock_representations.cpu().numpy()
        assert result_full["embeddings"] == expected_full.tolist()
        assert result_full["representation_type"] == "full"

    def test_get_sequence_representations_tensor_conversion(self):
        """Test proper tensor to list conversion for JSON serialization."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Create a specific tensor with known values
        mock_representations = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        mock_model.sequences_to_latents.return_value = [mock_representations]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )

        result = get_sequence_representations(request, mock_model_manager)

        # Verify the embeddings are Python lists, not numpy arrays
        assert isinstance(result["embeddings"], list)
        assert isinstance(result["embeddings"][0], list)
        assert isinstance(result["embeddings"][0][0], float)

        # Verify the values are correct
        expected_cls = mock_representations[:, 0, :].cpu().numpy()
        assert result["embeddings"] == expected_cls.tolist()

    def test_get_sequence_representations_multiple_layers(self):
        """Test that the last layer is used from sequences_to_latents."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock multiple layers, but only the last should be used
        mock_layer1 = torch.randn(1, 10, 8)
        mock_layer2 = torch.randn(1, 10, 8)
        mock_layer3 = torch.randn(1, 10, 8)
        mock_model.sequences_to_latents.return_value = [mock_layer1, mock_layer2, mock_layer3]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceRepresentationRequest(
            model_name="test_model",
            model_type="transformer",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            representation_type="cls",
        )

        result = get_sequence_representations(request, mock_model_manager)

        # Verify the last layer (mock_layer3) was used
        expected_cls = mock_layer3[:, 0, :].cpu().numpy()
        assert result["embeddings"] == expected_cls.tolist()
