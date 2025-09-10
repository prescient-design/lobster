"""Unit tests for representation tools in Lobster MCP server."""

import logging
from unittest.mock import Mock, patch

import pytest
import torch

from lobster.mcp.tools.representations import SequenceRepresentationResult, get_sequence_representations
from lobster.model import LobsterCBMPMLM, LobsterPMLM


class TestGetSequenceRepresentations:
    """Test the get_sequence_representations function."""

    def test_get_sequence_representations_masked_lm_pooled(self):
        """Test successful representation extraction from masked LM with pooled representation."""
        # Mock data
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"
        representation_type = "pooled"

        # Mock representations tensor (batch_size=1, seq_len=10, hidden_dim=512)
        mock_representations = torch.randn(1, 10, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type, representation_type)

        # Verify model method was called
        mock_model.sequences_to_latents.assert_called_once_with(sequences)

        # Verify result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 512
        assert result.embedding_dimension == 512
        assert result.num_sequences == 1
        assert result.representation_type == representation_type
        assert result.model_used == f"{model_type}_{model_name}"

    def test_get_sequence_representations_concept_bottleneck_cls(self):
        """Test successful representation extraction from concept bottleneck model with CLS representation."""
        # Mock data
        model_name = "cb_lobster_24M"
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "concept_bottleneck"
        representation_type = "cls"

        # Mock representations tensor (batch_size=2, seq_len=8, hidden_dim=256)
        mock_representations = torch.randn(2, 8, 256)

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type, representation_type)

        # Verify result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 256
        assert len(result.embeddings[1]) == 256
        assert result.embedding_dimension == 256
        assert result.num_sequences == 2
        assert result.representation_type == representation_type
        assert result.model_used == f"{model_type}_{model_name}"

    def test_get_sequence_representations_full_representation(self):
        """Test successful representation extraction with full sequence representation."""
        # Mock data
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"
        representation_type = "full"

        # Mock representations tensor (batch_size=1, seq_len=15, hidden_dim=512)
        mock_representations = torch.randn(1, 15, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type, representation_type)

        # Verify result structure for full representation
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 15  # seq_len
        assert len(result.embeddings[0][0]) == 512  # hidden_dim
        assert result.embedding_dimension == 512
        assert result.num_sequences == 1
        assert result.representation_type == representation_type
        assert result.model_used == f"{model_type}_{model_name}"

    def test_get_sequence_representations_default_parameters(self):
        """Test representation extraction with default parameters."""
        # Mock data
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        # Mock representations tensor
        mock_representations = torch.randn(1, 10, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type)

        # Verify default parameters were used
        assert result.representation_type == "pooled"
        assert result.embedding_dimension == 512

    def test_get_sequence_representations_model_loading_error(self):
        """Test error handling when model loading fails."""
        model_name = "nonexistent_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        with patch("lobster.mcp.tools.representations._load_model", side_effect=ValueError("Model not found")):
            with pytest.raises(ValueError, match="Model not found"):
                get_sequence_representations(model_name, sequences, model_type)

    def test_get_sequence_representations_inference_error(self):
        """Test error handling when inference fails."""
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        # Mock model that raises an exception during inference
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.side_effect = RuntimeError("Inference failed")

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            with pytest.raises(RuntimeError, match="Inference failed"):
                get_sequence_representations(model_name, sequences, model_type)

    def test_get_sequence_representations_invalid_representation_type(self):
        """Test error handling for invalid representation type."""
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"
        representation_type = "invalid_type"

        # Mock representations tensor
        mock_representations = torch.randn(1, 10, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            with pytest.raises(ValueError, match="Unknown representation type: invalid_type"):
                get_sequence_representations(model_name, sequences, model_type, representation_type)

    def test_get_sequence_representations_empty_sequences(self):
        """Test handling of empty sequences list."""
        model_name = "lobster_24M"
        sequences = []
        model_type = "masked_lm"

        # Mock representations tensor for empty batch
        mock_representations = torch.randn(0, 10, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type)

        # Verify result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 0
        assert result.embedding_dimension == 512
        assert result.num_sequences == 0
        assert result.representation_type == "pooled"
        assert result.model_used == f"{model_type}_{model_name}"

    def test_get_sequence_representations_single_empty_sequence(self):
        """Test handling of single empty sequence."""
        model_name = "lobster_24M"
        sequences = [""]
        model_type = "masked_lm"

        # Mock representations tensor for single empty sequence
        mock_representations = torch.randn(1, 2, 512)  # Empty sequence might have minimal tokens

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type)

        # Verify result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 512
        assert result.embedding_dimension == 512
        assert result.num_sequences == 1
        assert result.representation_type == "pooled"
        assert result.model_used == f"{model_type}_{model_name}"

    def test_get_sequence_representations_logging_on_error(self):
        """Test that errors are properly logged."""
        model_name = "test_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        with patch("lobster.mcp.tools.representations._load_model", side_effect=Exception("Test error")):
            with patch("lobster.mcp.tools.representations.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error"):
                    get_sequence_representations(model_name, sequences, model_type)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("Error getting representations: Test error")

    def test_get_sequence_representations_torch_no_grad_context(self):
        """Test that torch.no_grad() context is used for inference."""
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        # Mock representations tensor
        mock_representations = torch.randn(1, 10, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            with patch("lobster.mcp.tools.representations.torch.no_grad") as mock_no_grad:
                get_sequence_representations(model_name, sequences, model_type)

        # Verify torch.no_grad() was called
        mock_no_grad.assert_called_once()

    def test_get_sequence_representations_different_model_types(self):
        """Test representation extraction with different model types."""
        model_name = "test_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]

        # Test both model types
        for model_type in ["masked_lm", "concept_bottleneck"]:
            # Mock representations tensor
            mock_representations = torch.randn(1, 10, 256)

            # Mock model
            mock_model_class = LobsterPMLM if model_type == "masked_lm" else LobsterCBMPMLM
            mock_model = Mock(spec=mock_model_class)
            mock_model.sequences_to_latents.return_value = [mock_representations]

            with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
                result = get_sequence_representations(model_name, sequences, model_type)

            # Verify model type was handled correctly
            assert result.model_used == f"{model_type}_{model_name}"
            assert result.embedding_dimension == 256

    def test_get_sequence_representations_different_representation_types(self):
        """Test representation extraction with different representation types."""
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        # Test all representation types
        for representation_type in ["cls", "pooled", "full"]:
            # Mock representations tensor
            mock_representations = torch.randn(1, 10, 512)

            # Mock model
            mock_model = Mock(spec=LobsterPMLM)
            mock_model.sequences_to_latents.return_value = [mock_representations]

            with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
                result = get_sequence_representations(model_name, sequences, model_type, representation_type)

            # Verify representation type was handled correctly
            assert result.representation_type == representation_type
            assert result.embedding_dimension == 512


class TestRepresentationsIntegration:
    """Integration tests for representation functions."""

    def test_representation_workflow(self):
        """Test a complete representation extraction workflow."""
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "ACDEFGHIKLMNPQRSTVWY"]
        model_type = "masked_lm"
        representation_type = "pooled"

        # Mock representations tensor (batch_size=2, seq_len=12, hidden_dim=512)
        mock_representations = torch.randn(2, 12, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type, representation_type)

        # Verify complete result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 512
        assert len(result.embeddings[1]) == 512
        assert result.embedding_dimension == 512
        assert result.num_sequences == 2
        assert result.representation_type == representation_type
        assert result.model_used == f"{model_type}_{model_name}"

    def test_representation_with_realistic_data(self):
        """Test representation extraction with realistic protein sequence data."""
        model_name = "cb_lobster_24M"
        sequences = [
            "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"
        ]
        model_type = "concept_bottleneck"
        representation_type = "cls"

        # Mock representations tensor (batch_size=1, seq_len=20, hidden_dim=256)
        mock_representations = torch.randn(1, 20, 256)

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type, representation_type)

        # Verify result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 256
        assert result.embedding_dimension == 256
        assert result.num_sequences == 1
        assert result.representation_type == representation_type
        assert result.model_used == f"{model_type}_{model_name}"


class TestRepresentationsErrorHandling:
    """Test error handling and edge cases."""

    def test_get_sequence_representations_invalid_model_name(self):
        """Test handling of invalid model names."""
        model_name = ""
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        with patch("lobster.mcp.tools.representations._load_model", side_effect=ValueError("Invalid model name")):
            with pytest.raises(ValueError, match="Invalid model name"):
                get_sequence_representations(model_name, sequences, model_type)

    def test_get_sequence_representations_memory_error(self):
        """Test handling of memory errors during inference."""
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        # Mock model that raises a memory error
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.side_effect = RuntimeError("CUDA out of memory")

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                get_sequence_representations(model_name, sequences, model_type)

    def test_get_sequence_representations_tensor_conversion_error(self):
        """Test handling of tensor conversion errors."""
        model_name = "lobster_24M"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        # Create a real tensor that will fail when converted to list
        mock_representations = torch.randn(1, 10, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            with patch("torch.Tensor.cpu") as mock_cpu:
                # Mock the cpu() method to raise an exception
                mock_cpu.side_effect = Exception("Tensor conversion failed")

                with pytest.raises(Exception, match="Tensor conversion failed"):
                    get_sequence_representations(model_name, sequences, model_type)

    def test_get_sequence_representations_very_long_sequence(self):
        """Test handling of very long sequences."""
        model_name = "lobster_24M"
        # Create a very long sequence
        long_sequence = "A" * 10000
        sequences = [long_sequence]
        model_type = "masked_lm"

        # Mock representations tensor for long sequence
        mock_representations = torch.randn(1, 1000, 512)  # Truncated to max length

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type)

        # Verify result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 512
        assert result.embedding_dimension == 512
        assert result.num_sequences == 1

    def test_get_sequence_representations_mixed_sequence_lengths(self):
        """Test handling of sequences with mixed lengths."""
        model_name = "lobster_24M"
        sequences = ["A", "ACDEFGHIKLMNPQRSTVWY", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        # Mock representations tensor for mixed lengths (padded)
        mock_representations = torch.randn(3, 15, 512)

        # Mock model
        mock_model = Mock(spec=LobsterPMLM)
        mock_model.sequences_to_latents.return_value = [mock_representations]

        with patch("lobster.mcp.tools.representations._load_model", return_value=mock_model):
            result = get_sequence_representations(model_name, sequences, model_type)

        # Verify result structure
        assert isinstance(result, SequenceRepresentationResult)
        assert len(result.embeddings) == 3
        assert all(len(emb) == 512 for emb in result.embeddings)
        assert result.embedding_dimension == 512
        assert result.num_sequences == 3


class TestRepresentationsLogging:
    """Test logging behavior of representation functions."""

    def test_logging_configured(self):
        """Test that logging is properly configured."""
        from lobster.mcp.tools.representations import logger

        assert logger.name == "lobster-fastmcp-server"
        assert isinstance(logger, logging.Logger)

    def test_error_logging_format(self):
        """Test that error messages are logged with the correct format."""
        model_name = "test_model"
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        model_type = "masked_lm"

        with patch("lobster.mcp.tools.representations._load_model", side_effect=Exception("Test error message")):
            with patch("lobster.mcp.tools.representations.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error message"):
                    get_sequence_representations(model_name, sequences, model_type)

                # Verify the error message format
                mock_logger.error.assert_called_once_with("Error getting representations: Test error message")


class TestSequenceRepresentationResult:
    """Test the SequenceRepresentationResult dataclass."""

    def test_sequence_representation_result_creation(self):
        """Test creating a SequenceRepresentationResult instance."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embedding_dimension = 3
        num_sequences = 2
        representation_type = "pooled"
        model_used = "masked_lm_test_model"

        result = SequenceRepresentationResult(
            embeddings=embeddings,
            embedding_dimension=embedding_dimension,
            num_sequences=num_sequences,
            representation_type=representation_type,
            model_used=model_used,
        )

        assert result.embeddings == embeddings
        assert result.embedding_dimension == embedding_dimension
        assert result.num_sequences == num_sequences
        assert result.representation_type == representation_type
        assert result.model_used == model_used

    def test_sequence_representation_result_full_embeddings(self):
        """Test SequenceRepresentationResult with full sequence embeddings."""
        # Full embeddings: List[List[List[float]]] for full representation
        embeddings = [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # First sequence, all positions
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],  # Second sequence, all positions
        ]
        embedding_dimension = 3
        num_sequences = 2
        representation_type = "full"
        model_used = "masked_lm_test_model"

        result = SequenceRepresentationResult(
            embeddings=embeddings,
            embedding_dimension=embedding_dimension,
            num_sequences=num_sequences,
            representation_type=representation_type,
            model_used=model_used,
        )

        assert result.embeddings == embeddings
        assert result.embedding_dimension == embedding_dimension
        assert result.num_sequences == num_sequences
        assert result.representation_type == representation_type
        assert result.model_used == model_used

    def test_sequence_representation_result_empty_embeddings(self):
        """Test SequenceRepresentationResult with empty embeddings."""
        embeddings = []
        embedding_dimension = 512
        num_sequences = 0
        representation_type = "pooled"
        model_used = "masked_lm_test_model"

        result = SequenceRepresentationResult(
            embeddings=embeddings,
            embedding_dimension=embedding_dimension,
            num_sequences=num_sequences,
            representation_type=representation_type,
            model_used=model_used,
        )

        assert result.embeddings == embeddings
        assert result.embedding_dimension == embedding_dimension
        assert result.num_sequences == num_sequences
        assert result.representation_type == representation_type
        assert result.model_used == model_used
