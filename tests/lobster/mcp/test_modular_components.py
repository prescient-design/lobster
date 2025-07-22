"""
Tests for the modular MCP components
"""

import pytest
from unittest.mock import Mock, patch

try:
    from lobster.mcp.models import ModelManager, AVAILABLE_MODELS
    from lobster.mcp.schemas import (
        SequenceRepresentationRequest,
        SequenceConceptsRequest,
        InterventionRequest,
        SupportedConceptsRequest,
        NaturalnessRequest,
    )
    from lobster.mcp.tools import (
        get_sequence_representations,
        get_sequence_concepts,
        intervene_on_sequence,
        get_supported_concepts,
        compute_naturalness,
        list_available_models,
    )

    LOBSTER_AVAILABLE = True
except ImportError:
    LOBSTER_AVAILABLE = False


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestSchemas:
    """Test Pydantic request schemas"""

    def test_sequence_representation_request(self):
        """Test SequenceRepresentationRequest validation"""
        request = SequenceRepresentationRequest(
            sequences=["MKLLKN"], model_name="lobster_24M", model_type="masked_lm", representation_type="pooled"
        )

        assert request.sequences == ["MKLLKN"]
        assert request.model_name == "lobster_24M"
        assert request.model_type == "masked_lm"
        assert request.representation_type == "pooled"

    def test_sequence_concepts_request(self):
        """Test SequenceConceptsRequest validation"""
        request = SequenceConceptsRequest(sequences=["MKLLKN", "ACDEF"], model_name="cb_lobster_24M")

        assert request.sequences == ["MKLLKN", "ACDEF"]
        assert request.model_name == "cb_lobster_24M"

    def test_intervention_request(self):
        """Test InterventionRequest validation"""
        request = InterventionRequest(
            sequence="MKLLKN", concept="hydrophobic", model_name="cb_lobster_24M", edits=3, intervention_type="positive"
        )

        assert request.sequence == "MKLLKN"
        assert request.concept == "hydrophobic"
        assert request.edits == 3
        assert request.intervention_type == "positive"

    def test_intervention_request_defaults(self):
        """Test InterventionRequest default values"""
        request = InterventionRequest(sequence="MKLLKN", concept="hydrophobic", model_name="cb_lobster_24M")

        assert request.edits == 5  # default
        assert request.intervention_type == "negative"  # default

    def test_naturalness_request(self):
        """Test NaturalnessRequest validation"""
        request = NaturalnessRequest(sequences=["MKLLKN"], model_name="lobster_24M", model_type="masked_lm")

        assert request.sequences == ["MKLLKN"]
        assert request.model_name == "lobster_24M"
        assert request.model_type == "masked_lm"


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestTools:
    """Test MCP tool functions"""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager"""
        return Mock(spec=ModelManager)

    @pytest.fixture
    def mock_model(self):
        """Create a mock model"""
        model = Mock()
        model.sequences_to_latents.return_value = [Mock(), Mock()]  # Multiple layers
        return model

    def test_list_available_models(self, mock_model_manager):
        """Test list_available_models tool"""
        mock_model_manager.get_device_info.return_value = {"device": "cpu"}

        result = list_available_models(mock_model_manager)

        assert "available_models" in result
        assert "device" in result
        assert result["available_models"] == AVAILABLE_MODELS
        assert result["device"] == "cpu"

    @patch("lobster.mcp.tools.representations.torch")
    def test_get_sequence_representations(self, mock_torch, mock_model_manager, mock_model):
        """Test get_sequence_representations tool"""
        # Setup mocks
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.mean.return_value.cpu.return_value.numpy.return_value = [[0.1, 0.2, 0.3]]

        mock_model_manager.get_or_load_model.return_value = mock_model
        mock_model.sequences_to_latents.return_value = [
            Mock(),
            Mock(cpu=Mock(numpy=Mock(return_value=[[0.1, 0.2, 0.3]]))),
        ]

        request = SequenceRepresentationRequest(
            sequences=["MKLLKN"], model_name="lobster_24M", model_type="masked_lm", representation_type="pooled"
        )

        # Call the function
        result = get_sequence_representations(request, mock_model_manager)

        # Verify
        mock_model_manager.get_or_load_model.assert_called_once_with("lobster_24M", "masked_lm")
        assert "embeddings" in result
        assert "embedding_dimension" in result
        assert "num_sequences" in result
        assert result["num_sequences"] == 1

    def test_get_sequence_concepts(self, mock_model_manager):
        """Test get_sequence_concepts tool"""
        # Setup mock model
        mock_model = Mock()
        mock_concepts = Mock()
        mock_concepts.cpu.return_value.numpy.return_value.tolist.return_value = [[0.1, 0.2]]
        mock_concept_embeddings = Mock()
        mock_concept_embeddings.cpu.return_value.numpy.return_value.tolist.return_value = [[0.3, 0.4]]
        mock_concepts.shape = [1, 2]  # 1 sequence, 2 concepts

        mock_model.sequences_to_concepts.return_value = [Mock(), mock_concepts]
        mock_model.sequences_to_concepts_emb.return_value = [Mock(), mock_concept_embeddings]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SequenceConceptsRequest(sequences=["MKLLKN"], model_name="cb_lobster_24M")

        with patch("lobster.mcp.tools.concepts.torch"):
            result = get_sequence_concepts(request, mock_model_manager)

        mock_model_manager.get_or_load_model.assert_called_once_with("cb_lobster_24M", "concept_bottleneck")
        assert "concepts" in result
        assert "concept_embeddings" in result
        assert "num_concepts" in result
        assert result["num_sequences"] == 1

    def test_intervene_on_sequence(self, mock_model_manager):
        """Test intervene_on_sequence tool"""
        # Setup mock model
        mock_model = Mock()
        mock_model.intervene_on_sequences.return_value = ["MODIFIED_SEQUENCE"]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = InterventionRequest(
            sequence="ORIGINAL_SEQUENCE",
            concept="hydrophobic",
            model_name="cb_lobster_24M",
            edits=3,
            intervention_type="positive",
        )

        result = intervene_on_sequence(request, mock_model_manager)

        mock_model_manager.get_or_load_model.assert_called_once_with("cb_lobster_24M", "concept_bottleneck")
        mock_model.intervene_on_sequences.assert_called_once_with(
            ["ORIGINAL_SEQUENCE"], "hydrophobic", edits=3, intervention_type="positive"
        )

        assert result["original_sequence"] == "ORIGINAL_SEQUENCE"
        assert result["modified_sequence"] == "MODIFIED_SEQUENCE"
        assert result["concept"] == "hydrophobic"
        assert result["intervention_type"] == "positive"
        assert result["num_edits"] == 3

    def test_get_supported_concepts(self, mock_model_manager):
        """Test get_supported_concepts tool"""
        # Setup mock model
        mock_model = Mock()
        mock_model.list_supported_concept.return_value = ["concept1", "concept2"]
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = SupportedConceptsRequest(model_name="cb_lobster_24M")

        result = get_supported_concepts(request, mock_model_manager)

        mock_model_manager.get_or_load_model.assert_called_once_with("cb_lobster_24M", "concept_bottleneck")
        assert result["supported_concepts"] == ["concept1", "concept2"]
        assert result["num_concepts"] == 2

    @patch("lobster.mcp.tools.utils.torch")
    def test_compute_naturalness(self, mock_torch, mock_model_manager):
        """Test compute_naturalness tool"""
        # Setup mock model with naturalness method
        mock_model = Mock()
        mock_scores = Mock()
        mock_scores.tolist.return_value = [0.5, 0.7]
        mock_torch.is_tensor.return_value = True
        mock_model.naturalness.return_value = mock_scores
        mock_model_manager.get_or_load_model.return_value = mock_model

        request = NaturalnessRequest(sequences=["SEQ1", "SEQ2"], model_name="lobster_24M", model_type="masked_lm")

        result = compute_naturalness(request, mock_model_manager)

        mock_model_manager.get_or_load_model.assert_called_once_with("lobster_24M", "masked_lm")
        mock_model.naturalness.assert_called_once_with(["SEQ1", "SEQ2"])

        assert result["sequences"] == ["SEQ1", "SEQ2"]
        assert result["scores"] == [0.5, 0.7]
        assert result["model_used"] == "masked_lm_lobster_24M"


@pytest.mark.skipif(not LOBSTER_AVAILABLE, reason="Lobster not available")
class TestModelConfig:
    """Test model configuration"""

    def test_available_models_structure(self):
        """Test that AVAILABLE_MODELS has the expected structure"""
        assert isinstance(AVAILABLE_MODELS, dict)
        assert "masked_lm" in AVAILABLE_MODELS
        assert "concept_bottleneck" in AVAILABLE_MODELS

        # Check masked_lm models
        masked_lm = AVAILABLE_MODELS["masked_lm"]
        assert isinstance(masked_lm, dict)
        assert "lobster_24M" in masked_lm
        assert "lobster_150M" in masked_lm

        # Check concept_bottleneck models
        concept_bottleneck = AVAILABLE_MODELS["concept_bottleneck"]
        assert isinstance(concept_bottleneck, dict)
        assert "cb_lobster_24M" in concept_bottleneck
        assert "cb_lobster_150M" in concept_bottleneck
        assert "cb_lobster_650M" in concept_bottleneck
        assert "cb_lobster_3B" in concept_bottleneck

    def test_model_paths_format(self):
        """Test that model paths follow expected format"""
        for model_type, models in AVAILABLE_MODELS.items():
            for model_name, model_path in models.items():
                assert isinstance(model_path, str)
                assert "/" in model_path  # Should be in format "org/model"
