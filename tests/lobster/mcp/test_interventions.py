"""Tests for intervention-related tools in Lobster MCP server."""

import pytest
from unittest.mock import Mock, patch

try:
    from lobster.mcp.tools.interventions import intervene_on_sequence
    from lobster.mcp.schemas import InterventionRequest
    from lobster.mcp.models import ModelManager

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestInterveneOnSequence:
    """Test the intervene_on_sequence function."""

    def test_intervene_on_sequence_success(self):
        """Test successful concept intervention on a sequence."""
        # Setup mock model manager and model
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock intervention result
        mock_results = ["MODIFIED_SEQUENCE_ABC"]
        mock_model.intervene_on_sequences.return_value = mock_results
        mock_model_manager.get_or_load_model.return_value = mock_model

        # Create request
        request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE_XYZ",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        # Mock Levenshtein distance calculation
        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 5
            mock_import.return_value = mock_levenshtein_module

            # Call function
            result = intervene_on_sequence(request, mock_model_manager)

        # Verify model manager calls
        mock_model_manager.get_or_load_model.assert_called_once_with("test_model", "concept_bottleneck")

        # Verify model calls
        mock_model.intervene_on_sequences.assert_called_once_with(
            ["ORIGINAL_SEQUENCE_XYZ"], "hydrophobicity", edits=3, intervention_type="increase"
        )

        # Verify result structure
        assert "original_sequence" in result
        assert "modified_sequence" in result
        assert "concept" in result
        assert "intervention_type" in result
        assert "num_edits" in result
        assert "edit_distance" in result
        assert "model_used" in result

        # Verify result values
        assert result["original_sequence"] == "ORIGINAL_SEQUENCE_XYZ"
        assert result["modified_sequence"] == "MODIFIED_SEQUENCE_ABC"
        assert result["concept"] == "hydrophobicity"
        assert result["intervention_type"] == "increase"
        assert result["num_edits"] == 3
        assert result["edit_distance"] == 5
        assert result["model_used"] == "concept_bottleneck_test_model"

    def test_intervene_on_sequence_decrease_intervention(self):
        """Test concept intervention with decrease type."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_results = ["DECREASED_SEQUENCE"]
        mock_model.intervene_on_sequences.return_value = mock_results
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="stability",
            edits=2,
            intervention_type="decrease",
        )

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 3
            mock_import.return_value = mock_levenshtein_module

            result = intervene_on_sequence(request, mock_model_manager)

        assert result["intervention_type"] == "decrease"
        assert result["num_edits"] == 2
        assert result["concept"] == "stability"
        assert result["edit_distance"] == 3

    def test_intervene_on_sequence_default_parameters(self):
        """Test intervention with default parameters."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_results = ["DEFAULT_MODIFIED_SEQUENCE"]
        mock_model.intervene_on_sequences.return_value = mock_results
        mock_model_manager.get_or_load_model.return_value = mock_model

        # Use default values (edits=5, intervention_type="negative")
        request = InterventionRequest(model_name="test_model", sequence="ORIGINAL_SEQUENCE", concept="binding_site")

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 4
            mock_import.return_value = mock_levenshtein_module

            result = intervene_on_sequence(request, mock_model_manager)

        # Verify default values were used
        mock_model.intervene_on_sequences.assert_called_once_with(
            ["ORIGINAL_SEQUENCE"], "binding_site", edits=5, intervention_type="negative"
        )
        assert result["num_edits"] == 5
        assert result["intervention_type"] == "negative"

    def test_intervene_on_sequence_empty_results(self):
        """Test intervention when model returns empty results."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock empty results
        mock_model.intervene_on_sequences.return_value = []
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 0
            mock_import.return_value = mock_levenshtein_module

            result = intervene_on_sequence(request, mock_model_manager)

        assert result["modified_sequence"] is None
        assert result["edit_distance"] == 0

    def test_intervene_on_sequence_none_results(self):
        """Test intervention when model returns None results."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Mock None results
        mock_model.intervene_on_sequences.return_value = None
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 0
            mock_import.return_value = mock_levenshtein_module

            result = intervene_on_sequence(request, mock_model_manager)

        assert result["modified_sequence"] is None

    def test_intervene_on_sequence_levenshtein_import_error(self):
        """Test intervention when Levenshtein library is not available."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_results = ["MODIFIED_SEQUENCE"]
        mock_model.intervene_on_sequences.return_value = mock_results
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        # Mock ImportError for Levenshtein
        with patch("builtins.__import__", side_effect=ImportError("No module named 'Levenshtein'")):
            with patch("lobster.mcp.tools.interventions.logger") as mock_logger:
                result = intervene_on_sequence(request, mock_model_manager)

        # Verify warning was logged
        mock_logger.warning.assert_called_once_with("Levenshtein not available for edit distance calculation")

        # Verify edit_distance is None
        assert result["edit_distance"] is None
        assert result["modified_sequence"] == "MODIFIED_SEQUENCE"

    def test_intervene_on_sequence_model_loading_error(self):
        """Test handling of model loading errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Model not found")

        request = InterventionRequest(
            model_name="nonexistent_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        with pytest.raises(Exception, match="Model not found"):
            intervene_on_sequence(request, mock_model_manager)

    def test_intervene_on_sequence_intervention_error(self):
        """Test handling of intervention errors."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()
        mock_model.intervene_on_sequences.side_effect = Exception("Intervention failed")
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        with pytest.raises(Exception, match="Intervention failed"):
            intervene_on_sequence(request, mock_model_manager)

    def test_intervene_on_sequence_error_logging(self):
        """Test that errors are properly logged."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model_manager.get_or_load_model.side_effect = Exception("Test error")

        request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        with patch("lobster.mcp.tools.interventions.logger") as mock_logger:
            with pytest.raises(Exception, match="Test error"):
                intervene_on_sequence(request, mock_model_manager)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert "Error performing intervention" in mock_logger.error.call_args[0][0]

    def test_intervene_on_sequence_levenshtein_distance_calculation(self):
        """Test Levenshtein distance calculation with different scenarios."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        # Test with identical sequences
        mock_model.intervene_on_sequences.return_value = ["SAME_SEQUENCE"]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = InterventionRequest(
            model_name="test_model",
            sequence="SAME_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 0
            mock_import.return_value = mock_levenshtein_module

            result = intervene_on_sequence(request, mock_model_manager)

        assert result["edit_distance"] == 0

        # Test with completely different sequences
        mock_model.intervene_on_sequences.return_value = ["COMPLETELY_DIFFERENT"]

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 20
            mock_import.return_value = mock_levenshtein_module

            result = intervene_on_sequence(request, mock_model_manager)

        assert result["edit_distance"] == 20

    def test_intervene_on_sequence_request_validation(self):
        """Test that request objects are properly validated."""
        # Test valid request with all parameters
        valid_request = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=3,
            intervention_type="increase",
        )
        assert valid_request.model_name == "test_model"
        assert valid_request.sequence == "ORIGINAL_SEQUENCE"
        assert valid_request.concept == "hydrophobicity"
        assert valid_request.edits == 3
        assert valid_request.intervention_type == "increase"

        # Test valid request with defaults
        valid_request_defaults = InterventionRequest(
            model_name="test_model", sequence="ORIGINAL_SEQUENCE", concept="hydrophobicity"
        )
        assert valid_request_defaults.edits == 5  # default
        assert valid_request_defaults.intervention_type == "negative"  # default

    def test_intervene_on_sequence_multiple_edits(self):
        """Test intervention with different numbers of edits."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_results = ["MODIFIED_SEQUENCE"]
        mock_model.intervene_on_sequences.return_value = mock_results
        mock_model_manager.get_or_load_model.return_value = mock_model

        # Test with 1 edit
        request_1 = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=1,
            intervention_type="increase",
        )

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 1
            mock_import.return_value = mock_levenshtein_module

            result_1 = intervene_on_sequence(request_1, mock_model_manager)

        assert result_1["num_edits"] == 1
        assert result_1["edit_distance"] == 1

        # Test with 10 edits
        request_10 = InterventionRequest(
            model_name="test_model",
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobicity",
            edits=10,
            intervention_type="increase",
        )

        with patch("builtins.__import__") as mock_import:
            # Mock the Levenshtein module
            mock_levenshtein_module = Mock()
            mock_levenshtein_module.distance.return_value = 8
            mock_import.return_value = mock_levenshtein_module

            result_10 = intervene_on_sequence(request_10, mock_model_manager)

        assert result_10["num_edits"] == 10
        assert result_10["edit_distance"] == 8

    def test_intervene_on_sequence_different_concepts(self):
        """Test intervention with different biological concepts."""
        mock_model_manager = Mock(spec=ModelManager)
        mock_model = Mock()

        mock_results = ["MODIFIED_SEQUENCE"]
        mock_model.intervene_on_sequences.return_value = mock_results
        mock_model_manager.get_or_load_model.return_value = mock_model

        concepts_to_test = [
            "hydrophobicity",
            "secondary_structure",
            "binding_site",
            "active_site",
            "transmembrane_region",
            "stability",
            "flexibility",
        ]

        for concept in concepts_to_test:
            request = InterventionRequest(
                model_name="test_model",
                sequence="ORIGINAL_SEQUENCE",
                concept=concept,
                edits=3,
                intervention_type="increase",
            )

            with patch("builtins.__import__") as mock_import:
                # Mock the Levenshtein module
                mock_levenshtein_module = Mock()
                mock_levenshtein_module.distance.return_value = 3
                mock_import.return_value = mock_levenshtein_module

                result = intervene_on_sequence(request, mock_model_manager)

            assert result["concept"] == concept
            assert result["num_edits"] == 3
            assert result["intervention_type"] == "increase"
