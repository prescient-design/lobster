"""Tests for Pydantic schemas in Lobster MCP server."""

import pytest
from pydantic import ValidationError

try:
    from lobster.mcp.schemas.requests import (
        SequenceRepresentationRequest,
        SequenceConceptsRequest,
        InterventionRequest,
        SupportedConceptsRequest,
        NaturalnessRequest,
    )

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestSequenceRepresentationRequest:
    """Test SequenceRepresentationRequest model."""

    def test_valid_request_all_fields(self):
        """Test valid request with all fields specified."""
        request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test_model",
            model_type="masked_lm",
            representation_type="cls",
        )

        assert request.sequences == ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        assert request.model_name == "test_model"
        assert request.model_type == "masked_lm"
        assert request.representation_type == "cls"

    def test_valid_request_default_representation_type(self):
        """Test valid request with default representation_type."""
        request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test_model",
            model_type="concept_bottleneck",
        )

        assert request.representation_type == "pooled"  # default value

    def test_valid_request_multiple_sequences(self):
        """Test valid request with multiple sequences."""
        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "ACDEFGHIJKLMNOPQRSTUVWXYZ",
            "QWERTYUIOPASDFGHJKLZXCVBNM",
        ]

        request = SequenceRepresentationRequest(sequences=sequences, model_name="test_model", model_type="masked_lm")

        assert request.sequences == sequences
        assert len(request.sequences) == 3

    def test_valid_request_empty_sequences(self):
        """Test valid request with empty sequence list."""
        request = SequenceRepresentationRequest(sequences=[], model_name="test_model", model_type="masked_lm")

        assert request.sequences == []
        assert len(request.sequences) == 0

    def test_valid_request_different_representation_types(self):
        """Test valid request with different representation types."""
        representation_types = ["cls", "pooled", "full"]

        for rep_type in representation_types:
            request = SequenceRepresentationRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name="test_model",
                model_type="masked_lm",
                representation_type=rep_type,
            )
            assert request.representation_type == rep_type

    def test_valid_request_different_model_types(self):
        """Test valid request with different model types."""
        model_types = ["masked_lm", "concept_bottleneck"]

        for model_type in model_types:
            request = SequenceRepresentationRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name="test_model",
                model_type=model_type,
            )
            assert request.model_type == model_type

    def test_valid_request_special_characters_in_names(self):
        """Test valid request with special characters in model name."""
        request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test-model_1.0",
            model_type="masked_lm",
        )

        assert request.model_name == "test-model_1.0"

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing sequences
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(model_name="test_model", model_type="masked_lm")
        assert "sequences" in str(exc_info.value)

        # Missing model_name
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"], model_type="masked_lm"
            )
        assert "model_name" in str(exc_info.value)

        # Missing model_type
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"], model_name="test_model"
            )
        assert "model_type" in str(exc_info.value)

    def test_invalid_sequences_type(self):
        """Test that invalid sequences type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(
                sequences="not_a_list",  # Should be list
                model_name="test_model",
                model_type="masked_lm",
            )
        assert "sequences" in str(exc_info.value)

    def test_invalid_model_name_type(self):
        """Test that invalid model_name type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name=123,  # Should be string
                model_type="masked_lm",
            )
        assert "model_name" in str(exc_info.value)

    def test_invalid_model_type_type(self):
        """Test that invalid model_type type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name="test_model",
                model_type=456,  # Should be string
            )
        assert "model_type" in str(exc_info.value)

    def test_invalid_representation_type_type(self):
        """Test that invalid representation_type type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name="test_model",
                model_type="masked_lm",
                representation_type=789,  # Should be string
            )
        assert "representation_type" in str(exc_info.value)

    def test_empty_strings_in_sequences(self):
        """Test that empty strings in sequences are allowed."""
        request = SequenceRepresentationRequest(
            sequences=["", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", ""],
            model_name="test_model",
            model_type="masked_lm",
        )

        assert request.sequences == ["", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", ""]

    def test_empty_model_name(self):
        """Test that empty model name is allowed."""
        request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="",
            model_type="masked_lm",
        )

        assert request.model_name == ""

    def test_empty_model_type(self):
        """Test that empty model type is allowed."""
        request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test_model",
            model_type="",
        )

        assert request.model_type == ""

    def test_very_long_sequences(self):
        """Test that very long sequences are allowed."""
        long_sequence = "A" * 10000  # Very long sequence
        request = SequenceRepresentationRequest(
            sequences=[long_sequence], model_name="test_model", model_type="masked_lm"
        )

        assert request.sequences[0] == long_sequence
        assert len(request.sequences[0]) == 10000

    def test_very_long_model_name(self):
        """Test that very long model name is allowed."""
        long_name = "A" * 1000  # Very long name
        request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name=long_name,
            model_type="masked_lm",
        )

        assert request.model_name == long_name
        assert len(request.model_name) == 1000


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestSequenceConceptsRequest:
    """Test SequenceConceptsRequest model."""

    def test_valid_request(self):
        """Test valid request."""
        request = SequenceConceptsRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"], model_name="test_model"
        )

        assert request.sequences == ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        assert request.model_name == "test_model"

    def test_valid_request_multiple_sequences(self):
        """Test valid request with multiple sequences."""
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "ACDEFGHIJKLMNOPQRSTUVWXYZ"]

        request = SequenceConceptsRequest(sequences=sequences, model_name="test_model")

        assert request.sequences == sequences
        assert len(request.sequences) == 2

    def test_valid_request_empty_sequences(self):
        """Test valid request with empty sequence list."""
        request = SequenceConceptsRequest(sequences=[], model_name="test_model")

        assert request.sequences == []
        assert len(request.sequences) == 0

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing sequences
        with pytest.raises(ValidationError) as exc_info:
            SequenceConceptsRequest(model_name="test_model")
        assert "sequences" in str(exc_info.value)

        # Missing model_name
        with pytest.raises(ValidationError) as exc_info:
            SequenceConceptsRequest(sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"])
        assert "model_name" in str(exc_info.value)

    def test_invalid_sequences_type(self):
        """Test that invalid sequences type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SequenceConceptsRequest(
                sequences="not_a_list",  # Should be list
                model_name="test_model",
            )
        assert "sequences" in str(exc_info.value)

    def test_invalid_model_name_type(self):
        """Test that invalid model_name type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SequenceConceptsRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name=123,  # Should be string
            )
        assert "model_name" in str(exc_info.value)


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestInterventionRequest:
    """Test InterventionRequest model."""

    def test_valid_request_all_fields(self):
        """Test valid request with all fields specified."""
        request = InterventionRequest(
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            concept="hydrophobicity",
            model_name="test_model",
            edits=3,
            intervention_type="positive",
        )

        assert request.sequence == "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        assert request.concept == "hydrophobicity"
        assert request.model_name == "test_model"
        assert request.edits == 3
        assert request.intervention_type == "positive"

    def test_valid_request_default_values(self):
        """Test valid request with default values."""
        request = InterventionRequest(
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            concept="hydrophobicity",
            model_name="test_model",
        )

        assert request.edits == 5  # default value
        assert request.intervention_type == "negative"  # default value

    def test_valid_request_different_intervention_types(self):
        """Test valid request with different intervention types."""
        intervention_types = ["positive", "negative"]

        for int_type in intervention_types:
            request = InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                concept="hydrophobicity",
                model_name="test_model",
                intervention_type=int_type,
            )
            assert request.intervention_type == int_type

    def test_valid_request_different_edits_values(self):
        """Test valid request with different edits values."""
        edits_values = [0, 1, 5, 10, 100]

        for edits in edits_values:
            request = InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                concept="hydrophobicity",
                model_name="test_model",
                edits=edits,
            )
            assert request.edits == edits

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing sequence
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(concept="hydrophobicity", model_name="test_model")
        assert "sequence" in str(exc_info.value)

        # Missing concept
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", model_name="test_model"
            )
        assert "concept" in str(exc_info.value)

        # Missing model_name
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", concept="hydrophobicity"
            )
        assert "model_name" in str(exc_info.value)

    def test_invalid_sequence_type(self):
        """Test that invalid sequence type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(
                sequence=123,  # Should be string
                concept="hydrophobicity",
                model_name="test_model",
            )
        assert "sequence" in str(exc_info.value)

    def test_invalid_concept_type(self):
        """Test that invalid concept type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                concept=456,  # Should be string
                model_name="test_model",
            )
        assert "concept" in str(exc_info.value)

    def test_invalid_model_name_type(self):
        """Test that invalid model_name type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                concept="hydrophobicity",
                model_name=789,  # Should be string
            )
        assert "model_name" in str(exc_info.value)

    def test_invalid_edits_type(self):
        """Test that invalid edits type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                concept="hydrophobicity",
                model_name="test_model",
                edits="not_an_integer",  # Should be int
            )
        assert "edits" in str(exc_info.value)

    def test_invalid_intervention_type_type(self):
        """Test that invalid intervention_type type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InterventionRequest(
                sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
                concept="hydrophobicity",
                model_name="test_model",
                intervention_type=123,  # Should be string
            )
        assert "intervention_type" in str(exc_info.value)

    def test_empty_strings(self):
        """Test that empty strings are allowed."""
        request = InterventionRequest(sequence="", concept="", model_name="")

        assert request.sequence == ""
        assert request.concept == ""
        assert request.model_name == ""

    def test_negative_edits(self):
        """Test that negative edits are allowed."""
        request = InterventionRequest(
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            concept="hydrophobicity",
            model_name="test_model",
            edits=-5,
        )

        assert request.edits == -5


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestSupportedConceptsRequest:
    """Test SupportedConceptsRequest model."""

    def test_valid_request(self):
        """Test valid request."""
        request = SupportedConceptsRequest(model_name="test_model")

        assert request.model_name == "test_model"

    def test_valid_request_empty_model_name(self):
        """Test valid request with empty model name."""
        request = SupportedConceptsRequest(model_name="")

        assert request.model_name == ""

    def test_valid_request_special_characters(self):
        """Test valid request with special characters in model name."""
        request = SupportedConceptsRequest(model_name="test-model_1.0")

        assert request.model_name == "test-model_1.0"

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SupportedConceptsRequest()
        assert "model_name" in str(exc_info.value)

    def test_invalid_model_name_type(self):
        """Test that invalid model_name type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SupportedConceptsRequest(
                model_name=123  # Should be string
            )
        assert "model_name" in str(exc_info.value)

    def test_very_long_model_name(self):
        """Test that very long model name is allowed."""
        long_name = "A" * 1000  # Very long name
        request = SupportedConceptsRequest(model_name=long_name)

        assert request.model_name == long_name
        assert len(request.model_name) == 1000


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestNaturalnessRequest:
    """Test NaturalnessRequest model."""

    def test_valid_request(self):
        """Test valid request."""
        request = NaturalnessRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test_model",
            model_type="masked_lm",
        )

        assert request.sequences == ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
        assert request.model_name == "test_model"
        assert request.model_type == "masked_lm"

    def test_valid_request_multiple_sequences(self):
        """Test valid request with multiple sequences."""
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", "ACDEFGHIJKLMNOPQRSTUVWXYZ"]

        request = NaturalnessRequest(sequences=sequences, model_name="test_model", model_type="concept_bottleneck")

        assert request.sequences == sequences
        assert len(request.sequences) == 2

    def test_valid_request_empty_sequences(self):
        """Test valid request with empty sequence list."""
        request = NaturalnessRequest(sequences=[], model_name="test_model", model_type="masked_lm")

        assert request.sequences == []
        assert len(request.sequences) == 0

    def test_valid_request_different_model_types(self):
        """Test valid request with different model types."""
        model_types = ["masked_lm", "concept_bottleneck"]

        for model_type in model_types:
            request = NaturalnessRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name="test_model",
                model_type=model_type,
            )
            assert request.model_type == model_type

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing sequences
        with pytest.raises(ValidationError) as exc_info:
            NaturalnessRequest(model_name="test_model", model_type="masked_lm")
        assert "sequences" in str(exc_info.value)

        # Missing model_name
        with pytest.raises(ValidationError) as exc_info:
            NaturalnessRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"], model_type="masked_lm"
            )
        assert "model_name" in str(exc_info.value)

        # Missing model_type
        with pytest.raises(ValidationError) as exc_info:
            NaturalnessRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"], model_name="test_model"
            )
        assert "model_type" in str(exc_info.value)

    def test_invalid_sequences_type(self):
        """Test that invalid sequences type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            NaturalnessRequest(
                sequences="not_a_list",  # Should be list
                model_name="test_model",
                model_type="masked_lm",
            )
        assert "sequences" in str(exc_info.value)

    def test_invalid_model_name_type(self):
        """Test that invalid model_name type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            NaturalnessRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name=123,  # Should be string
                model_type="masked_lm",
            )
        assert "model_name" in str(exc_info.value)

    def test_invalid_model_type_type(self):
        """Test that invalid model_type type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            NaturalnessRequest(
                sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
                model_name="test_model",
                model_type=456,  # Should be string
            )
        assert "model_type" in str(exc_info.value)

    def test_empty_strings_in_sequences(self):
        """Test that empty strings in sequences are allowed."""
        request = NaturalnessRequest(
            sequences=["", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", ""],
            model_name="test_model",
            model_type="masked_lm",
        )

        assert request.sequences == ["", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG", ""]

    def test_empty_model_name(self):
        """Test that empty model name is allowed."""
        request = NaturalnessRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="",
            model_type="masked_lm",
        )

        assert request.model_name == ""

    def test_empty_model_type(self):
        """Test that empty model type is allowed."""
        request = NaturalnessRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test_model",
            model_type="",
        )

        assert request.model_type == ""


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestSchemaIntegration:
    """Integration tests for schemas."""

    def test_schema_serialization(self):
        """Test that schemas can be serialized to dict."""
        # Test SequenceRepresentationRequest
        rep_request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test_model",
            model_type="masked_lm",
            representation_type="cls",
        )
        rep_dict = rep_request.model_dump()
        assert "sequences" in rep_dict
        assert "model_name" in rep_dict
        assert "model_type" in rep_dict
        assert "representation_type" in rep_dict

        # Test InterventionRequest
        int_request = InterventionRequest(
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            concept="hydrophobicity",
            model_name="test_model",
            edits=3,
            intervention_type="positive",
        )
        int_dict = int_request.model_dump()
        assert "sequence" in int_dict
        assert "concept" in int_dict
        assert "model_name" in int_dict
        assert "edits" in int_dict
        assert "intervention_type" in int_dict

    def test_schema_json_serialization(self):
        """Test that schemas can be serialized to JSON."""
        request = SequenceConceptsRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"], model_name="test_model"
        )

        json_str = request.model_dump_json()
        assert isinstance(json_str, str)
        assert "sequences" in json_str
        assert "test_model" in json_str

    def test_schema_field_descriptions(self):
        """Test that field descriptions are accessible."""
        # Check that field descriptions exist
        sequence_field = SequenceRepresentationRequest.model_fields["sequences"]
        assert sequence_field.description == "List of protein sequences"

        model_name_field = SequenceRepresentationRequest.model_fields["model_name"]
        assert model_name_field.description == "Name of the model to use"

        representation_type_field = SequenceRepresentationRequest.model_fields["representation_type"]
        assert representation_type_field.description == "Type of representation: 'cls', 'pooled', or 'full'"

    def test_schema_default_values(self):
        """Test that default values are correctly set."""
        # Test default representation_type
        request = SequenceRepresentationRequest(
            sequences=["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
            model_name="test_model",
            model_type="masked_lm",
        )
        assert request.representation_type == "pooled"

        # Test default edits
        request = InterventionRequest(
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            concept="hydrophobicity",
            model_name="test_model",
        )
        assert request.edits == 5

        # Test default intervention_type
        assert request.intervention_type == "negative"

    def test_schema_validation_error_details(self):
        """Test that validation errors provide detailed information."""
        with pytest.raises(ValidationError) as exc_info:
            SequenceRepresentationRequest(sequences="not_a_list", model_name="test_model", model_type="masked_lm")

        # Check that the error contains useful information
        error_str = str(exc_info.value)
        assert "sequences" in error_str
        assert "Input should be a valid list" in error_str or "Input should be a valid array" in error_str
