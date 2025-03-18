from unittest.mock import MagicMock, patch

import pytest
import torch
from lobster.model._ume import Ume


@pytest.fixture
def smiles_examples():
    return ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


@pytest.fixture
def protein_examples():
    return ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"]


@pytest.fixture
def dna_examples():
    return ["ATGCATGC", "GCTAGCTA"]


class TestUme:
    """Tests for the Universal Molecular Encoder (Ume) class"""

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_frozen_parameters(self, mock_load_checkpoint):
        """Test that parameters are frozen when freeze=True"""
        # Create mock model
        mock_model = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_model.model.parameters.return_value = mock_params
        mock_load_checkpoint.return_value = mock_model

        # Create Ume with frozen parameters
        ume = Ume("dummy_checkpoint.ckpt", freeze=True)

        # Verify that load_from_checkpoint was called
        mock_load_checkpoint.assert_called_once_with("dummy_checkpoint.ckpt")

        # Verify that parameters were accessed
        mock_model.model.parameters.assert_called()

        # Verify freeze attribute is True
        assert ume.freeze is True

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_unfrozen_parameters(self, mock_load_checkpoint):
        """Test that parameters are not frozen when freeze=False"""
        # Create mock model
        mock_model = MagicMock()
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        mock_model.model.parameters.return_value = mock_params
        mock_load_checkpoint.return_value = mock_model

        # Create Ume without freezing parameters
        ume = Ume("dummy_checkpoint.ckpt", freeze=False)

        # Verify freeze attribute is False
        assert ume.freeze is False

        # Verify that parameters were not frozen
        mock_model.model.parameters.assert_not_called()

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    @patch("lobster.model._ume.UmeSmilesTokenizerFast")
    @patch("lobster.model._ume.UmeAminoAcidTokenizerFast")
    @patch("lobster.model._ume.UmeNucleotideTokenizerFast")
    def test_get_tokenizer(self, mock_nucleotide, mock_amino, mock_smiles, mock_load_checkpoint):
        """Test getting tokenizers for different modalities"""
        # Set up model mock
        mock_model = MagicMock()
        mock_load_checkpoint.return_value = mock_model

        # Setup tokenizer mocks
        mock_smiles_instance = MagicMock()
        mock_amino_instance = MagicMock()
        mock_nucleotide_instance = MagicMock()

        mock_smiles.return_value = mock_smiles_instance
        mock_amino.return_value = mock_amino_instance
        mock_nucleotide.return_value = mock_nucleotide_instance

        # Create Ume instance
        ume = Ume("dummy_checkpoint.ckpt")

        # Test each modality
        tokenizer_map = {
            "SMILES": (mock_smiles, mock_smiles_instance),
            "amino_acid": (mock_amino, mock_amino_instance),
            "nucleotide": (mock_nucleotide, mock_nucleotide_instance),
        }

        for modality, (mock_cls, mock_instance) in tokenizer_map.items():
            # Get tokenizer
            tokenizer = ume.get_tokenizer(["test"], modality)

            # Verify correct tokenizer class was instantiated
            mock_cls.assert_called_once()

            # Verify the returned tokenizer is our mock instance
            assert tokenizer == mock_instance

            # Reset mock for next test
            mock_cls.reset_mock()

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_get_embeddings_basic(self, mock_load_checkpoint, smiles_examples, protein_examples, dna_examples):
        """Test basic embedding functionality for all modalities"""
        # Mock model with controlled output
        mock_model = MagicMock()
        mock_model.max_length = 512
        mock_model.device = torch.device("cpu")

        # Set up tokens_to_latents to return predictable tensor
        def mock_tokens_to_latents(**kwargs):
            batch_size = kwargs["input_ids"].size(0)
            seq_len = kwargs["input_ids"].size(1)
            hidden_size = 768
            return torch.ones(batch_size * seq_len, hidden_size)

        mock_model.tokens_to_latents = mock_tokens_to_latents
        mock_load_checkpoint.return_value = mock_model

        # Create Ume instance
        ume = Ume("dummy_checkpoint.ckpt")

        # Test for each modality
        modalities = ["SMILES", "amino_acid", "nucleotide"]
        test_inputs = {"SMILES": smiles_examples, "amino_acid": protein_examples, "nucleotide": dna_examples}

        for modality in modalities:
            # Mock tokenizer for this modality
            mock_tokenizer = MagicMock()

            # Configure tokenizer to return input tensors
            batch_size = len(test_inputs[modality])
            seq_len = 10  # Small sequence length for test
            mock_tokenizer.return_value = {
                "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, seq_len),
            }

            # Patch get_tokenizer to return our mock
            with patch.object(ume, "get_tokenizer", return_value=mock_tokenizer):
                # Test aggregated embeddings
                embeddings = ume.get_embeddings(test_inputs[modality], modality)
                assert embeddings.shape == (batch_size, 768)

                # Test token-level embeddings
                token_embeddings = ume.get_embeddings(test_inputs[modality], modality, aggregate=False)
                assert token_embeddings.shape == (batch_size, seq_len, 768)
