"""Unit tests for concept-related tools in Lobster MCP server."""

import logging
from unittest.mock import Mock, patch

import pytest
import torch

from lobster.mcp.tools.concepts import get_sequence_concepts, get_supported_concepts
from lobster.model import LobsterCBMPMLM


class TestGetSequenceConcepts:
    """Test the get_sequence_concepts function."""

    def test_get_sequence_concepts_success(self):
        """Test successful concept prediction for sequences."""
        # Mock data
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Mock concept predictions and embeddings
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.7]])
        mock_concept_embeddings = torch.tensor([[0.5, 0.3, 0.8, 0.2, 0.6]])

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_sequence_concepts(model_name, sequences)

        # Verify model methods were called
        mock_model.sequences_to_concepts.assert_called_once_with(sequences)
        mock_model.sequences_to_concepts_emb.assert_called_once_with(sequences)

        # Verify result structure
        assert len(result["concepts"]) == 1
        assert len(result["concepts"][0]) == 5
        assert len(result["concept_embeddings"]) == 1
        assert len(result["concept_embeddings"][0]) == 5
        # Check approximate equality for floating point values
        assert abs(result["concepts"][0][0] - 0.8) < 1e-6
        assert abs(result["concepts"][0][1] - 0.2) < 1e-6
        assert abs(result["concepts"][0][2] - 0.9) < 1e-6
        assert abs(result["concepts"][0][3] - 0.1) < 1e-6
        assert abs(result["concepts"][0][4] - 0.7) < 1e-6
        assert abs(result["concept_embeddings"][0][0] - 0.5) < 1e-6
        assert abs(result["concept_embeddings"][0][1] - 0.3) < 1e-6
        assert abs(result["concept_embeddings"][0][2] - 0.8) < 1e-6
        assert abs(result["concept_embeddings"][0][3] - 0.2) < 1e-6
        assert abs(result["concept_embeddings"][0][4] - 0.6) < 1e-6
        assert result["num_sequences"] == 1
        assert result["num_concepts"] == 5
        assert result["model_used"] == f"concept_bottleneck_{model_name}"

    def test_get_sequence_concepts_multiple_sequences(self):
        """Test concept prediction for multiple sequences."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "ACDEFGHIKLMNPQRSTVWY"]

        # Mock concept predictions and embeddings for multiple sequences
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.7], [0.3, 0.7, 0.4, 0.8, 0.2]])
        mock_concept_embeddings = torch.tensor([[0.5, 0.3, 0.8, 0.2, 0.6], [0.1, 0.9, 0.4, 0.7, 0.3]])

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_sequence_concepts(model_name, sequences)

        # Verify result structure
        assert len(result["concepts"]) == 2
        assert len(result["concept_embeddings"]) == 2
        assert result["num_sequences"] == 2
        assert result["num_concepts"] == 5
        assert result["model_used"] == f"concept_bottleneck_{model_name}"

    def test_get_sequence_concepts_empty_sequences(self):
        """Test concept prediction with empty sequence list."""
        model_name = "test_concept_model"
        sequences = []

        # Mock empty concept predictions and embeddings
        mock_concepts = torch.tensor([])
        mock_concept_embeddings = torch.tensor([])

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_sequence_concepts(model_name, sequences)

        # Verify result structure
        assert result["concepts"] == []
        assert result["concept_embeddings"] == []
        assert result["num_sequences"] == 0
        assert result["num_concepts"] == 0
        assert result["model_used"] == f"concept_bottleneck_{model_name}"

    def test_get_sequence_concepts_model_loading_error(self):
        """Test error handling when model loading fails."""
        model_name = "nonexistent_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=ValueError("Model not found")):
            with pytest.raises(ValueError, match="Model not found"):
                get_sequence_concepts(model_name, sequences)

    def test_get_sequence_concepts_model_inference_error(self):
        """Test error handling when model inference fails."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Mock model that raises an exception during inference
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.side_effect = RuntimeError("CUDA out of memory")

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                get_sequence_concepts(model_name, sequences)

    def test_get_sequence_concepts_torch_no_grad_context(self):
        """Test that torch.no_grad() context is used for inference."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Mock concept predictions and embeddings
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.7]])
        mock_concept_embeddings = torch.tensor([[0.5, 0.3, 0.8, 0.2, 0.6]])

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        # Mock torch.no_grad to verify it's called
        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            with patch("torch.no_grad") as mock_no_grad:
                mock_no_grad.return_value.__enter__ = Mock()
                mock_no_grad.return_value.__exit__ = Mock()

                get_sequence_concepts(model_name, sequences)

                # Verify torch.no_grad was called
                mock_no_grad.assert_called_once()

    def test_get_sequence_concepts_tensor_conversion(self):
        """Test that tensors are properly converted to CPU and then to lists."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Create tensors on CPU (simulating GPU tensors that need conversion)
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.7]])
        mock_concept_embeddings = torch.tensor([[0.5, 0.3, 0.8, 0.2, 0.6]])

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_sequence_concepts(model_name, sequences)

        # Verify tensors were converted to lists
        assert isinstance(result["concepts"], list)
        assert isinstance(result["concept_embeddings"], list)
        assert len(result["concepts"]) == 1
        assert len(result["concepts"][0]) == 5
        assert len(result["concept_embeddings"]) == 1
        assert len(result["concept_embeddings"][0]) == 5
        # Check approximate equality for floating point values
        assert abs(result["concepts"][0][0] - 0.8) < 1e-6
        assert abs(result["concepts"][0][1] - 0.2) < 1e-6
        assert abs(result["concepts"][0][2] - 0.9) < 1e-6
        assert abs(result["concepts"][0][3] - 0.1) < 1e-6
        assert abs(result["concepts"][0][4] - 0.7) < 1e-6
        assert abs(result["concept_embeddings"][0][0] - 0.5) < 1e-6
        assert abs(result["concept_embeddings"][0][1] - 0.3) < 1e-6
        assert abs(result["concept_embeddings"][0][2] - 0.8) < 1e-6
        assert abs(result["concept_embeddings"][0][3] - 0.2) < 1e-6
        assert abs(result["concept_embeddings"][0][4] - 0.6) < 1e-6

    def test_get_sequence_concepts_logging_on_error(self):
        """Test that errors are properly logged."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=Exception("Test error")):
            with patch("lobster.mcp.tools.concepts.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error"):
                    get_sequence_concepts(model_name, sequences)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("Error getting concepts: Test error")


class TestGetSupportedConcepts:
    """Test the get_supported_concepts function."""

    def test_get_supported_concepts_success_list(self):
        """Test successful retrieval of supported concepts as a list."""
        model_name = "test_concept_model"
        concepts_list = ["hydrophobicity", "secondary_structure", "binding_site", "active_site"]

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = concepts_list

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_supported_concepts(model_name)

        # Verify model method was called
        mock_model.list_supported_concept.assert_called_once()

        # Verify result structure
        assert result["concepts"] == concepts_list
        assert result["model_name"] == model_name
        assert result["num_concepts"] == 4

    def test_get_supported_concepts_none_return(self):
        """Test handling when model returns None for concepts."""
        model_name = "test_concept_model"

        # Mock model that returns None
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = None

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_supported_concepts(model_name)

        # Verify result structure
        assert result["concepts"] == []
        assert result["model_name"] == model_name
        assert result["num_concepts"] == 0

    def test_get_supported_concepts_string_return(self):
        """Test handling when model returns a string instead of a list."""
        model_name = "test_concept_model"
        concept_string = "hydrophobicity"

        # Mock model that returns a string
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = concept_string

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_supported_concepts(model_name)

        # Verify result structure
        assert result["concepts"] == [concept_string]
        assert result["model_name"] == model_name
        assert result["num_concepts"] == 1

    def test_get_supported_concepts_empty_string_return(self):
        """Test handling when model returns an empty string."""
        model_name = "test_concept_model"

        # Mock model that returns an empty string
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = ""

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_supported_concepts(model_name)

        # Verify result structure
        assert result["concepts"] == []
        assert result["model_name"] == model_name
        assert result["num_concepts"] == 0

    def test_get_supported_concepts_empty_list_return(self):
        """Test handling when model returns an empty list."""
        model_name = "test_concept_model"

        # Mock model that returns an empty list
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = []

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_supported_concepts(model_name)

        # Verify result structure
        assert result["concepts"] == []
        assert result["model_name"] == model_name
        assert result["num_concepts"] == 0

    def test_get_supported_concepts_model_loading_error(self):
        """Test error handling when model loading fails."""
        model_name = "nonexistent_model"

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=ValueError("Model not found")):
            with pytest.raises(ValueError, match="Model not found"):
                get_supported_concepts(model_name)

    def test_get_supported_concepts_model_method_error(self):
        """Test error handling when model method fails."""
        model_name = "test_concept_model"

        # Mock model that raises an exception
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.side_effect = AttributeError("Method not found")

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            with pytest.raises(AttributeError, match="Method not found"):
                get_supported_concepts(model_name)

    def test_get_supported_concepts_logging_on_error(self):
        """Test that errors are properly logged."""
        model_name = "test_concept_model"

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=Exception("Test error")):
            with patch("lobster.mcp.tools.concepts.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error"):
                    get_supported_concepts(model_name)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("Error getting supported concepts: Test error")

    def test_get_supported_concepts_non_list_return(self):
        """Test handling when model returns a non-list, non-string, non-None value."""
        model_name = "test_concept_model"

        # Mock model that returns a number
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = 42

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_supported_concepts(model_name)

        # Verify result structure - should wrap the value in a list
        assert result["concepts"] == [42]
        assert result["model_name"] == model_name
        assert result["num_concepts"] == 1

    def test_get_supported_concepts_zero_return(self):
        """Test handling when model returns zero."""
        model_name = "test_concept_model"

        # Mock model that returns zero
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = 0

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_supported_concepts(model_name)

        # Verify result structure - 0 is falsy, so it should be treated as empty
        assert result["concepts"] == []
        assert result["model_name"] == model_name
        assert result["num_concepts"] == 0


class TestConceptsIntegration:
    """Integration tests for concept-related functions."""

    def test_concepts_workflow(self):
        """Test a complete workflow: get supported concepts, then get concepts for sequences."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        concepts_list = ["hydrophobicity", "secondary_structure", "binding_site"]

        # Mock concept predictions and embeddings
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9]])
        mock_concept_embeddings = torch.tensor([[0.5, 0.3, 0.8]])

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = concepts_list
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            # First, get supported concepts
            supported_result = get_supported_concepts(model_name)

            # Then, get concepts for sequences
            concepts_result = get_sequence_concepts(model_name, sequences)

        # Verify supported concepts result
        assert supported_result["concepts"] == concepts_list
        assert supported_result["num_concepts"] == 3

        # Verify concepts result
        assert len(concepts_result["concepts"]) == 1
        assert len(concepts_result["concepts"][0]) == 3
        assert len(concepts_result["concept_embeddings"]) == 1
        assert len(concepts_result["concept_embeddings"][0]) == 3
        # Check approximate equality for floating point values
        assert abs(concepts_result["concepts"][0][0] - 0.8) < 1e-6
        assert abs(concepts_result["concepts"][0][1] - 0.2) < 1e-6
        assert abs(concepts_result["concepts"][0][2] - 0.9) < 1e-6
        assert abs(concepts_result["concept_embeddings"][0][0] - 0.5) < 1e-6
        assert abs(concepts_result["concept_embeddings"][0][1] - 0.3) < 1e-6
        assert abs(concepts_result["concept_embeddings"][0][2] - 0.8) < 1e-6
        assert concepts_result["num_concepts"] == 3
        assert concepts_result["num_sequences"] == 1

    def test_concepts_with_realistic_data(self):
        """Test with realistic protein sequence data."""
        model_name = "cb_lobster_24M"
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "ACDEFGHIKLMNPQRSTVWY",
            "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF",
        ]

        # Mock realistic concept predictions (718 concepts as per cb_lobster_24M)
        num_concepts = 718
        mock_concepts = torch.randn(len(sequences), num_concepts)
        mock_concept_embeddings = torch.randn(len(sequences), num_concepts)

        # Mock realistic concept names
        concepts_list = [f"concept_{i}" for i in range(num_concepts)]

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.list_supported_concept.return_value = concepts_list
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            # Get supported concepts
            supported_result = get_supported_concepts(model_name)

            # Get concepts for sequences
            concepts_result = get_sequence_concepts(model_name, sequences)

        # Verify supported concepts result
        assert len(supported_result["concepts"]) == num_concepts
        assert supported_result["num_concepts"] == num_concepts
        assert supported_result["model_name"] == model_name

        # Verify concepts result
        assert len(concepts_result["concepts"]) == len(sequences)
        assert len(concepts_result["concept_embeddings"]) == len(sequences)
        assert concepts_result["num_concepts"] == num_concepts
        assert concepts_result["num_sequences"] == len(sequences)
        assert concepts_result["model_used"] == f"concept_bottleneck_{model_name}"

        # Verify each sequence has the correct number of concepts
        for concept_pred in concepts_result["concepts"]:
            assert len(concept_pred) == num_concepts

        for concept_emb in concepts_result["concept_embeddings"]:
            assert len(concept_emb) == num_concepts


class TestConceptsErrorHandling:
    """Test error handling and edge cases."""

    def test_get_sequence_concepts_invalid_model_name(self):
        """Test handling of invalid model names."""
        model_name = ""
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=ValueError("Invalid model name")):
            with pytest.raises(ValueError, match="Invalid model name"):
                get_sequence_concepts(model_name, sequences)

    def test_get_sequence_concepts_invalid_sequences(self):
        """Test handling of invalid sequence data."""
        model_name = "test_concept_model"
        sequences = [123, 456]  # Invalid: should be strings

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.side_effect = TypeError("Sequences must be strings")

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            with pytest.raises(TypeError, match="Sequences must be strings"):
                get_sequence_concepts(model_name, sequences)

    def test_get_supported_concepts_invalid_model_name(self):
        """Test handling of invalid model names in get_supported_concepts."""
        model_name = ""

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=ValueError("Invalid model name")):
            with pytest.raises(ValueError, match="Invalid model name"):
                get_supported_concepts(model_name)

    def test_get_sequence_concepts_memory_error(self):
        """Test handling of memory errors during inference."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Mock model that raises a memory error
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA out of memory"):
                get_sequence_concepts(model_name, sequences)

    def test_get_sequence_concepts_tensor_shape_mismatch(self):
        """Test handling of tensor shape mismatches."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Mock tensors with mismatched shapes
        mock_concepts = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.7]])
        mock_concept_embeddings = torch.tensor([[0.5, 0.3, 0.8]])  # Different shape!

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_concepts.return_value = [mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [mock_concept_embeddings]

        with patch("lobster.mcp.tools.concepts._load_model", return_value=mock_model):
            result = get_sequence_concepts(model_name, sequences)

        # The function should still work, but the shapes will be different
        assert len(result["concepts"][0]) == 5
        assert len(result["concept_embeddings"][0]) == 3
        assert result["num_concepts"] == 5  # Based on concepts tensor


class TestConceptsLogging:
    """Test logging behavior of concept functions."""

    def test_logging_configured(self):
        """Test that logging is properly configured."""
        from lobster.mcp.tools.concepts import logger

        assert logger.name == "lobster-fastmcp-server"
        assert isinstance(logger, logging.Logger)

    def test_error_logging_format(self):
        """Test that error messages are logged with the correct format."""
        model_name = "test_concept_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=Exception("Test error message")):
            with patch("lobster.mcp.tools.concepts.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error message"):
                    get_sequence_concepts(model_name, sequences)

                # Verify the error message format
                mock_logger.error.assert_called_once_with("Error getting concepts: Test error message")

    def test_supported_concepts_error_logging_format(self):
        """Test that error messages are logged with the correct format for get_supported_concepts."""
        model_name = "test_concept_model"

        with patch("lobster.mcp.tools.concepts._load_model", side_effect=Exception("Test error message")):
            with patch("lobster.mcp.tools.concepts.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error message"):
                    get_supported_concepts(model_name)

                # Verify the error message format
                mock_logger.error.assert_called_once_with("Error getting supported concepts: Test error message")
