"""Tests for concept-related tools in Lobster MCP server."""

import pytest
import torch
from unittest.mock import Mock, patch

try:
    from lobster.mcp.tools.concepts import get_sequence_concepts, get_supported_concepts
    from lobster.mcp.schemas import SequenceConceptsRequest, SupportedConceptsRequest
    from lobster.mcp.models import ModelManager

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestGetSequenceConcepts:
    """Test the get_sequence_concepts function."""

    def test_get_sequence_concepts_success(self):
        """Test successful concept prediction for sequences."""
        # Setup mock model manager and model
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock concept predictions and embeddings
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.7], [0.3, 0.8, 0.4, 0.6, 0.2]])
        mock_embeddings = torch.tensor([[0.5, 0.3, 0.8, 0.2, 0.6], [0.7, 0.1, 0.9, 0.4, 0.3]])

        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_embeddings]
        mock_model_manager.get_or_load_model.return_value = mock_model

        # Create request
        request = SequenceConceptsRequest(
            model_name="test_model",
            sequences=[
                "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                "ACDEFGHIJKLMNOPQRSTUVWXYZ",
            ],
        )

        # Call function
        result = get_sequence_concepts(request, mock_model_manager)

        # Verify model manager calls
        mock_model_manager.get_or_load_model.assert_called_once_with("test_model", "concept_bottleneck")

        # Verify model calls
        mock_model.sequences_to_concepts.assert_called_once_with(request.sequences)
        mock_model.sequences_to_concepts_emb.assert_called_once_with(request.sequences)

        # Verify result structure
        assert "concepts" in result
        assert "concept_embeddings" in result
        assert "num_sequences" in result
        assert "num_concepts" in result
        assert "model_used" in result

        # Verify result values
        assert result["num_sequences"] == 2
        assert result["num_concepts"] == 5
        assert result["model_used"] == "concept_bottleneck_test_model"

        # Verify concepts and embeddings are lists
        assert isinstance(result["concepts"], list)
        assert isinstance(result["concept_embeddings"], list)
        assert len(result["concepts"]) == 2
        assert len(result["concept_embeddings"]) == 2

        # Verify tensor conversion
        assert result["concepts"] == mock_concepts.cpu().numpy().tolist()
        assert result["concept_embeddings"] == mock_embeddings.cpu().numpy().tolist()

    def test_get_sequence_concepts_single_sequence(self):
        """Test concept prediction for a single sequence."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_concepts = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.7]])
        mock_embeddings = torch.tensor([[0.5, 0.3, 0.8, 0.2, 0.6]])

        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_embeddings]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceConceptsRequest(
            model_name="test_model", sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        )

        result = get_sequence_concepts(request, mock_model_manager)

        assert result["num_sequences"] == 1
        assert result["num_concepts"] == 5
        assert len(result["concepts"]) == 1
        assert len(result["concept_embeddings"]) == 1

    def test_get_sequence_concepts_empty_sequences(self):
        """Test concept prediction with empty sequence list."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_concepts = torch.tensor([])
        mock_embeddings = torch.tensor([])

        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_embeddings]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceConceptsRequest(model_name="test_model", sequences=[])

        result = get_sequence_concepts(request, mock_model_manager)

        assert result["num_sequences"] == 0
        assert result["num_concepts"] == 0
        assert result["concepts"] == []
        assert result["concept_embeddings"] == []

    def test_get_sequence_concepts_model_loading_error(self):
        """Test handling of model loading errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Model not found")

        request = SequenceConceptsRequest(
            model_name="nonexistent_model",
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
        )

        with pytest.raises(Exception, match="Model not found"):
            get_sequence_concepts(request, mock_model_manager)

    def test_get_sequence_concepts_model_inference_error(self):
        """Test handling of model inference errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()
        mock_model.sequences_to_concepts.side_effect = Exception("Inference failed")
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceConceptsRequest(
            model_name="test_model", sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        )

        with pytest.raises(Exception, match="Inference failed"):
            get_sequence_concepts(request, mock_model_manager)

    def test_get_sequence_concepts_torch_no_grad_context(self):
        """Test that torch.no_grad() is used during inference."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_concepts = torch.tensor([[0.8, 0.2, 0.9]])
        mock_embeddings = torch.tensor([[0.5, 0.3, 0.8]])

        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_embeddings]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceConceptsRequest(
            model_name="test_model", sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        )

        # Mock torch.no_grad to verify it's called
        with patch("torch.no_grad") as mock_no_grad:
            mock_no_grad.return_value.__enter__ = Mock()
            mock_no_grad.return_value.__exit__ = Mock()

            get_sequence_concepts(request, mock_model_manager)

            # Verify torch.no_grad was called
            mock_no_grad.assert_called_once()


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestGetSupportedConcepts:
    """Test the get_supported_concepts function."""

    def test_get_supported_concepts_success_list(self):
        """Test successful retrieval of supported concepts as a list."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_concepts = ["hydrophobicity", "secondary_structure", "binding_site", "active_site", "transmembrane_region"]
        mock_model.list_supported_concept.return_value = mock_concepts
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SupportedConceptsRequest(model_name="test_model")

        result = get_supported_concepts(request, mock_model_manager)

        # Verify model manager calls
        mock_model_manager.get_or_load_model.assert_called_once_with("test_model", "concept_bottleneck")
        mock_model.list_supported_concept.assert_called_once()

        # Verify result structure
        assert "supported_concepts" in result
        assert "num_concepts" in result
        assert "model_used" in result

        # Verify result values
        assert result["supported_concepts"] == mock_concepts
        assert result["num_concepts"] == 5
        assert result["model_used"] == "concept_bottleneck_test_model"

    def test_get_supported_concepts_empty_list(self):
        """Test retrieval of supported concepts when list is empty."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_model.list_supported_concept.return_value = []
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SupportedConceptsRequest(model_name="test_model")

        result = get_supported_concepts(request, mock_model_manager)

        assert result["supported_concepts"] == []
        assert result["num_concepts"] == 0
        assert result["model_used"] == "concept_bottleneck_test_model"

    def test_get_supported_concepts_non_list_return(self):
        """Test handling when model returns non-list concepts."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock model returning a string instead of a list
        mock_model.list_supported_concept.return_value = "concept_string"
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SupportedConceptsRequest(model_name="test_model")

        result = get_supported_concepts(request, mock_model_manager)

        assert result["supported_concepts"] == "concept_string"
        assert result["num_concepts"] is None
        assert result["model_used"] == "concept_bottleneck_test_model"

    def test_get_supported_concepts_none_return(self):
        """Test handling when model returns None for concepts."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_model.list_supported_concept.return_value = None
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SupportedConceptsRequest(model_name="test_model")

        result = get_supported_concepts(request, mock_model_manager)

        assert result["supported_concepts"] is None
        assert result["num_concepts"] is None
        assert result["model_used"] == "concept_bottleneck_test_model"

    def test_get_supported_concepts_model_loading_error(self):
        """Test handling of model loading errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Model not found")

        request = SupportedConceptsRequest(model_name="nonexistent_model")

        with pytest.raises(Exception, match="Model not found"):
            get_supported_concepts(request, mock_model_manager)

    def test_get_supported_concepts_method_error(self):
        """Test handling of errors in list_supported_concept method."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()
        mock_model.list_supported_concept.side_effect = Exception("Method failed")
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SupportedConceptsRequest(model_name="test_model")

        with pytest.raises(Exception, match="Method failed"):
            get_supported_concepts(request, mock_model_manager)

    def test_get_supported_concepts_large_list(self):
        """Test retrieval of a large list of concepts."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Create a large list of mock concepts
        mock_concepts = [f"concept_{i}" for i in range(1000)]
        mock_model.list_supported_concept.return_value = mock_concepts
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SupportedConceptsRequest(model_name="test_model")

        result = get_supported_concepts(request, mock_model_manager)

        assert result["supported_concepts"] == mock_concepts
        assert result["num_concepts"] == 1000
        assert len(result["supported_concepts"]) == 1000


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestConceptsIntegration:
    """Integration tests for concept-related functions."""

    def test_concepts_workflow(self):
        """Test a complete workflow: get supported concepts, then get sequence concepts."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Setup mock for supported concepts
        mock_concepts_list = ["hydrophobicity", "secondary_structure", "binding_site"]
        mock_model.list_supported_concept.return_value = mock_concepts_list

        # Setup mock for sequence concepts
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9]])
        mock_embeddings = torch.tensor([[0.5, 0.3, 0.8]])
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_embeddings]

        mock_model_manager.get_or_load_model.return_value = mock_model

        # Step 1: Get supported concepts
        supported_request = SupportedConceptsRequest(model_name="test_model")
        supported_result = get_supported_concepts(supported_request, mock_model_manager)

        assert supported_result["num_concepts"] == 3
        assert "hydrophobicity" in supported_result["supported_concepts"]

        # Step 2: Get sequence concepts
        sequence_request = SequenceConceptsRequest(
            model_name="test_model", sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        )
        sequence_result = get_sequence_concepts(sequence_request, mock_model_manager)

        assert sequence_result["num_concepts"] == 3
        assert sequence_result["num_sequences"] == 1

        # Verify the same model was used for both calls
        assert supported_result["model_used"] == sequence_result["model_used"]

    def test_concepts_error_logging(self):
        """Test that errors are properly logged."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Test error")

        request = SequenceConceptsRequest(
            model_name="test_model", sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        )

        with patch("lobster.mcp.tools.concepts.logger") as mock_logger:
            with pytest.raises(Exception, match="Test error"):
                get_sequence_concepts(request, mock_model_manager)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Error getting concepts" in mock_logger.error.call_args[0][0]

    def test_concepts_request_validation(self):
        """Test that request objects are properly validated."""
        # Test valid request
        valid_request = SequenceConceptsRequest(
            model_name="test_model", sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        )
        assert valid_request.model_name == "test_model"
        assert len(valid_request.sequences) == 1

        # Test valid supported concepts request
        valid_supported_request = SupportedConceptsRequest(model_name="test_model")
        assert valid_supported_request.model_name == "test_model"
