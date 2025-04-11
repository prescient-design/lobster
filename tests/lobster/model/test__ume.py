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
        "3d_coordinates": ["aa", "bb"],
    }


class TestUme:
    """Tests for the Universal Molecular Encoder (Ume) class"""

    @patch("lobster.tokenization.UmeTokenizerTransform")
    def test_initialization(self, mock_transform):
        """Test basic initialization of Ume class"""
        # Mock the tokenizer to avoid actual model initialization
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.mask_token_id = 1
        mock_tokenizer.cls_token_id = 2
        mock_tokenizer.eos_token_id = 3

        # Return the mock tokenizer from the transform
        mock_transform_instance = MagicMock()
        mock_transform_instance.tokenizer = mock_tokenizer
        mock_transform.return_value = mock_transform_instance

        # Mock FlexBERT to avoid actual model initialization
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume(model_name="UME_mini")

            # Verify transform initialization for each modality
            assert len(ume.tokenizer_transforms) == len(Modality)

            # Check that each modality has a transform
            for modality in Modality:
                assert modality in ume.tokenizer_transforms

            # Check initial state
            assert ume.frozen is False

    @patch("lobster.tokenization.UmeTokenizerTransform")
    def test_get_tokenizer(self, mock_transform):
        """Test getting tokenizer for different modalities"""
        # Create mock tokenizers
        mock_tokenizers = {}
        for modality in Modality:
            mock_tokenizer = MagicMock()
            mock_tokenizers[modality] = mock_tokenizer

            # Configure the transform instance for this modality
            mock_transform_instance = MagicMock()
            mock_transform_instance.tokenizer = mock_tokenizer
            mock_transform.return_value = mock_transform_instance

        # Mock FlexBERT to avoid model initialization
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume()

            # Replace tokenizer_transforms with our mocks
            for modality in Modality:
                transform = MagicMock()
                transform.tokenizer = mock_tokenizers[modality]
                ume.tokenizer_transforms[modality] = transform

            # Test with string modality
            for modality_str in ["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]:
                tokenizer = ume.get_tokenizer(modality_str)
                modality_enum = Modality(modality_str)
                assert tokenizer == mock_tokenizers[modality_enum]

            # Test with Modality enum
            for modality_enum in Modality:
                tokenizer = ume.get_tokenizer(modality_enum)
                assert tokenizer == mock_tokenizers[modality_enum]

    def test_get_embeddings(self, sample_sequences):
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

        # Mock dependencies to avoid actual initialization
        with (
            patch("lobster.tokenization.UmeTokenizerTransform", MagicMock()),
            patch("lobster.model._ume.FlexBERT", MagicMock()),
        ):
            ume = Ume()
            ume.model = mock_model

            # Test for each modality
            for modality, inputs in sample_sequences.items():
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

                # Replace the transform
                modality_enum = Modality(modality)
                ume.tokenizer_transforms[modality_enum] = mock_transform

                # Test aggregated embeddings
                embeddings = ume.get_embeddings(inputs, modality)
                assert embeddings.shape == (batch_size, 768)

                # Test token-level embeddings
                token_embeddings = ume.get_embeddings(inputs, modality, aggregate=False)
                assert token_embeddings.shape == (batch_size, seq_len, 768)

                # Test using Modality enum directly
                embeddings_enum = ume.get_embeddings(inputs, modality_enum)
                assert embeddings_enum.shape == (batch_size, 768)

    def test_get_vocab(self):
        """Test get_vocab method"""
        # Mock dependencies to avoid actual initialization
        with (
            patch("lobster.tokenization.UmeTokenizerTransform", MagicMock()),
            patch("lobster.model._ume.FlexBERT", MagicMock()),
        ):
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
        # Mock dependencies to avoid actual initialization
        with (
            patch("lobster.tokenization.UmeTokenizerTransform", MagicMock()),
            patch("lobster.model._ume.FlexBERT", MagicMock()),
        ):
            ume = Ume()
            ume.model = None  # Explicitly set model to None

            with pytest.raises(ValueError):
                ume.get_embeddings(["test"], "SMILES")

    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing model parameters"""
        # Create mock model with mock parameters
        mock_model = MagicMock()
        mock_params = [MagicMock(), MagicMock()]
        mock_model.parameters.return_value = mock_params

        # Mock dependencies to avoid actual initialization
        with (
            patch("lobster.tokenization.UmeTokenizerTransform", MagicMock()),
            patch("lobster.model._ume.FlexBERT", MagicMock()),
        ):
            ume = Ume()
            ume.model = mock_model

            # Test freeze
            ume.freeze()
            assert ume.frozen is True
            mock_model.eval.assert_called_once()
            for param in mock_params:
                assert param.requires_grad is False

            # Test unfreeze
            ume.unfreeze()
            assert ume.frozen is False
            mock_model.train.assert_called_once()
            for param in mock_params:
                assert param.requires_grad is True

    def test_configure_optimizers(self):
        """Test configure_optimizers method"""
        # Create mock model that returns a specific optimizer config
        mock_model = MagicMock()
        expected_config = {"optimizer": MagicMock(), "lr_scheduler": {"scheduler": MagicMock(), "interval": "step"}}
        mock_model.configure_optimizers.return_value = expected_config

        # Mock dependencies to avoid actual initialization
        with (
            patch("lobster.tokenization.UmeTokenizerTransform", MagicMock()),
            patch("lobster.model._ume.FlexBERT", MagicMock()),
        ):
            ume = Ume()
            ume.model = mock_model

            # Test the method
            config = ume.configure_optimizers()
            assert config == expected_config
            mock_model.configure_optimizers.assert_called_once()
