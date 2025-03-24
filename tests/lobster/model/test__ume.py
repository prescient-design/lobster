from unittest.mock import MagicMock, patch

import pytest
import torch
from lobster.constants import Modality
from lobster.model._ume import Ume


@pytest.fixture
def sample_sequences():
    return {
        "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"],
        "nucleotide": ["ATGCATGC", "GCTAGCTA"],
    }


class TestUme:
    """Tests for the Universal Molecular Encoder (Ume) class"""

    @patch("lobster.tokenization.UmeTokenizerTransform")
    def test_initialization(self, mock_transform):
        """Test basic initialization of Ume class"""
        ume = Ume()

        # Verify transform initialization for each modality
        assert len(ume.tokenizer_transforms) == len(Modality)

        # Check that each modality has a transform
        for modality in Modality:
            assert modality in ume.tokenizer_transforms

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_load_from_checkpoint(self, mock_load_checkpoint):
        """Test loading model from checkpoint with various options"""
        mock_model = MagicMock()
        mock_model.max_length = 512
        mock_model.model = MagicMock()
        mock_load_checkpoint.return_value = mock_model

        # Test with default freeze=True
        ume = Ume.load_from_checkpoint("dummy_checkpoint.ckpt")
        assert ume.model == mock_model
        assert ume.freeze is True

        # Model parameters should be frozen
        for param in mock_model.model.parameters.return_value:
            assert param.requires_grad is False

        # Test with freeze=False
        ume = Ume.load_from_checkpoint("dummy_checkpoint.ckpt", freeze=False)
        assert ume.freeze is False

    def test_modalities_property(self):
        """Test modalities property returns correct values"""
        ume = Ume()
        expected_modalities = [modality.value for modality in Modality]
        assert sorted(ume.modalities) == sorted(expected_modalities)

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_max_length_property(self, mock_load_checkpoint):
        """Test max_length property returns model's max_length"""
        mock_model = MagicMock()
        mock_model.max_length = 512
        mock_load_checkpoint.return_value = mock_model

        ume = Ume.load_from_checkpoint("dummy_checkpoint.ckpt")
        assert ume.max_length == 512

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_get_tokenizer(self, mock_load_checkpoint):
        """Test getting tokenizer for different modalities"""
        ume = Ume()

        # Create mock transforms
        for modality in Modality:
            transform = MagicMock()
            transform.tokenizer = MagicMock()
            ume.tokenizer_transforms[modality] = transform

        # Check each modality
        for modality_str in ["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]:
            tokenizer = ume.get_tokenizer(modality_str)
            modality_enum = Modality(modality_str)
            assert tokenizer == ume.tokenizer_transforms[modality_enum].tokenizer

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_get_embeddings(self, mock_load_checkpoint, sample_sequences):
        """Test get_embeddings for different modalities"""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")

        # Mock tokens_to_latents to return predictable tensor
        def mock_tokens_to_latents(**kwargs):
            batch_size = kwargs["input_ids"].size(0)
            seq_len = kwargs["input_ids"].size(1)
            hidden_size = 768
            return torch.ones(batch_size * seq_len, hidden_size)

        mock_model.tokens_to_latents = mock_tokens_to_latents
        mock_load_checkpoint.return_value = mock_model

        # Create Ume instance
        ume = Ume.load_from_checkpoint("dummy_checkpoint.ckpt")
        ume.model = mock_model

        # Test for one modality (SMILES)
        modality = "SMILES"
        inputs = sample_sequences[modality]

        # Create mock transform
        mock_transform = MagicMock()
        batch_size = len(inputs)
        seq_len = 10

        # Configure transform return value
        mock_transform.return_value = {
            "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, seq_len),
            "modality_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
        }

        # Patch the transform
        modality_enum = Modality(modality)
        with patch.dict(ume.tokenizer_transforms, {modality_enum: mock_transform}):
            # Test aggregated embeddings
            embeddings = ume.get_embeddings(inputs, modality)
            assert embeddings.shape == (batch_size, 768)

            # Test token-level embeddings
            token_embeddings = ume.get_embeddings(inputs, modality, aggregate=False)
            assert token_embeddings.shape == (batch_size, seq_len, 768)

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_get_vocab(self, mock_load_checkpoint):
        """Test get_vocab method"""
        ume = Ume()

        # Create two mock tokenizers with different vocabs
        tokenizer1 = MagicMock()
        tokenizer1.get_vocab.return_value = {"token1": 1, "token2": 2, "reserved_token": 3}

        tokenizer2 = MagicMock()
        tokenizer2.get_vocab.return_value = {"token3": 4, "token4": 5, "reserved_special": 6}

        # Create transforms with the mock tokenizers
        for i, modality in enumerate(Modality):
            mock_transform = MagicMock()
            mock_transform.tokenizer = tokenizer1 if i % 2 == 0 else tokenizer2
            ume.tokenizer_transforms[modality] = mock_transform

        # Get vocabulary and check result
        vocab = ume.get_vocab()
        expected_vocab = {1: "token1", 2: "token2", 4: "token3", 5: "token4"}
        assert vocab == expected_vocab

    def test_get_embeddings_without_model(self):
        """Test error handling when get_embeddings is called without model"""
        ume = Ume()  # No model loaded

        with pytest.raises(ValueError):
            ume.get_embeddings(["test"], "SMILES")
