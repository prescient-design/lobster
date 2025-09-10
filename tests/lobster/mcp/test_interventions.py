"""Unit tests for intervention tools in Lobster MCP server."""

import logging
from unittest.mock import Mock, patch

import pytest

from lobster.mcp.tools.interventions import intervene_on_sequence
from lobster.model import LobsterCBMPMLM


class TestInterveneOnSequence:
    """Test the intervene_on_sequence function."""

    def test_intervene_on_sequence_success_negative(self):
        """Test successful negative intervention on a sequence."""
        # Mock data
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"
        edits = 3
        intervention_type = "negative"

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 2
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept, edits, intervention_type)

        # Verify model method was called
        mock_model.intervene_on_sequences.assert_called_once_with(
            [sequence], concept, edits=edits, intervention_type=intervention_type
        )

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == intervention_type
        assert result["num_edits"] == edits
        assert result["edit_distance"] == 2
        assert result["model_used"] == f"concept_bottleneck_{model_name}"

    def test_intervene_on_sequence_success_positive(self):
        """Test successful positive intervention on a sequence."""
        # Mock data
        model_name = "test_concept_model"
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        concept = "stability"
        edits = 5
        intervention_type = "positive"

        # Mock modified sequence
        modified_sequence = "ACDEFGHIKLMNPQRSTVWY"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 0
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept, edits, intervention_type)

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == intervention_type
        assert result["num_edits"] == edits
        assert result["edit_distance"] == 0
        assert result["model_used"] == f"concept_bottleneck_{model_name}"

    def test_intervene_on_sequence_default_parameters(self):
        """Test intervention with default parameters."""
        # Mock data
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "gravy"

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 1
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept)

        # Verify default parameters were used
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 5
        assert result["edit_distance"] == 1

    def test_intervene_on_sequence_model_loading_error(self):
        """Test error handling when model loading fails."""
        model_name = "nonexistent_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        with patch("lobster.mcp.tools.interventions._load_model", side_effect=ValueError("Model not found")):
            with pytest.raises(ValueError, match="Model not found"):
                intervene_on_sequence(model_name, sequence, concept)

    def test_intervene_on_sequence_intervention_error(self):
        """Test error handling when intervention fails."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock model that raises an exception during intervention
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.side_effect = RuntimeError("Intervention failed")

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with pytest.raises(RuntimeError, match="Intervention failed"):
                intervene_on_sequence(model_name, sequence, concept)

    def test_intervene_on_sequence_empty_results(self):
        """Test handling when intervention returns empty results."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock model that returns empty results
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = []

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 0
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept)

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] is None
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 5
        assert result["edit_distance"] == 0

    def test_intervene_on_sequence_none_results(self):
        """Test handling when intervention returns None results."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock model that returns None results
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = None

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 0
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept)

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] is None
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 5
        assert result["edit_distance"] == 0

    def test_intervene_on_sequence_levenshtein_import_error(self):
        """Test handling when Levenshtein library is not available."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'Levenshtein'")):
                with patch("lobster.mcp.tools.interventions.logger") as mock_logger:
                    result = intervene_on_sequence(model_name, sequence, concept)

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 5
        assert result["edit_distance"] is None

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("Levenshtein not available for edit distance calculation")

    def test_intervene_on_sequence_levenshtein_calculation_error(self):
        """Test handling when Levenshtein distance calculation fails."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import that raises an exception during distance calculation
                mock_levenshtein = Mock()
                mock_levenshtein.distance.side_effect = Exception("Distance calculation failed")
                mock_import.return_value = mock_levenshtein

                # The function should raise the exception since it only catches ImportError
                with pytest.raises(Exception, match="Distance calculation failed"):
                    intervene_on_sequence(model_name, sequence, concept)

    def test_intervene_on_sequence_logging_on_error(self):
        """Test that errors are properly logged."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        with patch("lobster.mcp.tools.interventions._load_model", side_effect=Exception("Test error")):
            with patch("lobster.mcp.tools.interventions.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error"):
                    intervene_on_sequence(model_name, sequence, concept)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("Error performing intervention: Test error")

    def test_intervene_on_sequence_different_edit_counts(self):
        """Test intervention with different edit counts."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Test different edit counts
        for edits in [1, 3, 5, 10]:
            # Mock modified sequence
            modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

            # Mock model
            mock_model = Mock(spec=LobsterCBMPMLM)
            mock_model.intervene_on_sequences.return_value = [modified_sequence]

            with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
                with patch("builtins.__import__") as mock_import:
                    # Mock Levenshtein import
                    mock_levenshtein = Mock()
                    mock_levenshtein.distance.return_value = edits
                    mock_import.return_value = mock_levenshtein

                    result = intervene_on_sequence(model_name, sequence, concept, edits=edits)

            # Verify edit count was passed correctly
            assert result["num_edits"] == edits
            mock_model.intervene_on_sequences.assert_called_with(
                [sequence], concept, edits=edits, intervention_type="negative"
            )

    def test_intervene_on_sequence_different_intervention_types(self):
        """Test intervention with different intervention types."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Test both intervention types
        for intervention_type in ["positive", "negative"]:
            # Mock modified sequence
            modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

            # Mock model
            mock_model = Mock(spec=LobsterCBMPMLM)
            mock_model.intervene_on_sequences.return_value = [modified_sequence]

            with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
                with patch("builtins.__import__") as mock_import:
                    # Mock Levenshtein import
                    mock_levenshtein = Mock()
                    mock_levenshtein.distance.return_value = 1
                    mock_import.return_value = mock_levenshtein

                    result = intervene_on_sequence(model_name, sequence, concept, intervention_type=intervention_type)

            # Verify intervention type was passed correctly
            assert result["intervention_type"] == intervention_type
            mock_model.intervene_on_sequences.assert_called_with(
                [sequence], concept, edits=5, intervention_type=intervention_type
            )


class TestInterventionsIntegration:
    """Integration tests for intervention functions."""

    def test_intervention_workflow(self):
        """Test a complete intervention workflow."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 3
                mock_import.return_value = mock_levenshtein

                # Perform intervention
                result = intervene_on_sequence(model_name, sequence, concept, edits=3, intervention_type="negative")

        # Verify complete result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 3
        assert result["edit_distance"] == 3
        assert result["model_used"] == f"concept_bottleneck_{model_name}"

    def test_intervention_with_realistic_data(self):
        """Test intervention with realistic protein sequence data."""
        model_name = "cb_lobster_24M"
        sequence = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"
        concept = "gravy"

        # Mock modified sequence (slightly different)
        modified_sequence = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 5
                mock_import.return_value = mock_levenshtein

                # Perform intervention
                result = intervene_on_sequence(model_name, sequence, concept, edits=5, intervention_type="negative")

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 5
        assert result["edit_distance"] == 5
        assert result["model_used"] == f"concept_bottleneck_{model_name}"


class TestInterventionsErrorHandling:
    """Test error handling and edge cases."""

    def test_intervene_on_sequence_invalid_model_name(self):
        """Test handling of invalid model names."""
        model_name = ""
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        with patch("lobster.mcp.tools.interventions._load_model", side_effect=ValueError("Invalid model name")):
            with pytest.raises(ValueError, match="Invalid model name"):
                intervene_on_sequence(model_name, sequence, concept)

    def test_intervene_on_sequence_empty_sequence(self):
        """Test handling of empty sequence."""
        model_name = "test_concept_model"
        sequence = ""
        concept = "hydrophobicity"

        # Mock modified sequence
        modified_sequence = ""

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 0
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept)

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 5
        assert result["edit_distance"] == 0

    def test_intervene_on_sequence_empty_concept(self):
        """Test handling of empty concept."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = ""

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 0
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept)

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == 5
        assert result["edit_distance"] == 0

    def test_intervene_on_sequence_zero_edits(self):
        """Test handling of zero edits."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"
        edits = 0

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__") as mock_import:
                # Mock Levenshtein import
                mock_levenshtein = Mock()
                mock_levenshtein.distance.return_value = 0
                mock_import.return_value = mock_levenshtein

                result = intervene_on_sequence(model_name, sequence, concept, edits=edits)

        # Verify result structure
        assert result["original_sequence"] == sequence
        assert result["modified_sequence"] == modified_sequence
        assert result["concept"] == concept
        assert result["intervention_type"] == "negative"
        assert result["num_edits"] == edits
        assert result["edit_distance"] == 0

    def test_intervene_on_sequence_memory_error(self):
        """Test handling of memory errors during intervention."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock model that raises a memory error
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.side_effect = RuntimeError("CUDA out of memory")

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                intervene_on_sequence(model_name, sequence, concept)

    def test_intervene_on_sequence_concept_not_found(self):
        """Test handling when concept is not found in model."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "nonexistent_concept"

        # Mock model that raises an exception for unknown concept
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.side_effect = ValueError("Concept not found")

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with pytest.raises(ValueError, match="Concept not found"):
                intervene_on_sequence(model_name, sequence, concept)


class TestInterventionsLogging:
    """Test logging behavior of intervention functions."""

    def test_logging_configured(self):
        """Test that logging is properly configured."""
        from lobster.mcp.tools.interventions import logger

        assert logger.name == "lobster-fastmcp-server"
        assert isinstance(logger, logging.Logger)

    def test_error_logging_format(self):
        """Test that error messages are logged with the correct format."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        with patch("lobster.mcp.tools.interventions._load_model", side_effect=Exception("Test error message")):
            with patch("lobster.mcp.tools.interventions.logger") as mock_logger:
                with pytest.raises(Exception, match="Test error message"):
                    intervene_on_sequence(model_name, sequence, concept)

                # Verify the error message format
                mock_logger.error.assert_called_once_with("Error performing intervention: Test error message")

    def test_levenshtein_warning_logging(self):
        """Test that Levenshtein import warnings are logged correctly."""
        model_name = "test_concept_model"
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        concept = "hydrophobicity"

        # Mock modified sequence
        modified_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

        # Mock model
        mock_model = Mock(spec=LobsterCBMPMLM)
        mock_model.intervene_on_sequences.return_value = [modified_sequence]

        with patch("lobster.mcp.tools.interventions._load_model", return_value=mock_model):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'Levenshtein'")):
                with patch("lobster.mcp.tools.interventions.logger") as mock_logger:
                    result = intervene_on_sequence(model_name, sequence, concept)

                    # Verify warning was logged
                    mock_logger.warning.assert_called_once_with(
                        "Levenshtein not available for edit distance calculation"
                    )

                    # Verify result still works
                    assert result["edit_distance"] is None
