from unittest.mock import MagicMock, patch

import pytest
import torch
from lobster.constants import Modality
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

    @patch("lobster.model._ume.UmeSmilesTokenizerFast")
    @patch("lobster.model._ume.UmeAminoAcidTokenizerFast")
    @patch("lobster.model._ume.UmeNucleotideTokenizerFast")
    @patch("lobster.model._ume.UmeLatentGenerator3DCoordTokenizerFast")
    def test_tokenizer_initialization(self, mock_coord, mock_nucleotide, mock_amino, mock_smiles):
        """Test that tokenizers are initialized during __init__"""
        # Setup tokenizer mocks
        mock_smiles_instance = MagicMock()
        mock_amino_instance = MagicMock()
        mock_nucleotide_instance = MagicMock()
        mock_coord_instance = MagicMock()

        mock_smiles.return_value = mock_smiles_instance
        mock_amino.return_value = mock_amino_instance
        mock_nucleotide.return_value = mock_nucleotide_instance
        mock_coord.return_value = mock_coord_instance

        # Create Ume instance
        ume = Ume()

        mock_smiles.assert_called_once()
        mock_amino.assert_called_once()
        mock_nucleotide.assert_called_once()
        mock_coord.assert_called_once()

        assert ume.tokenizers[Modality.SMILES] == mock_smiles_instance
        assert ume.tokenizers[Modality.AMINO_ACID] == mock_amino_instance
        assert ume.tokenizers[Modality.NUCLEOTIDE] == mock_nucleotide_instance
        assert ume.tokenizers[Modality.COORDINATES_3D] == mock_coord_instance

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_get_tokenizer(self, mock_load_checkpoint):
        """Test getting tokenizers for different modalities"""
        ume = Ume()

        mock_tokenizers = {}
        for modality in Modality:
            mock_tokenizers[modality] = MagicMock()

        ume.tokenizers = mock_tokenizers

        modality_map = {
            "SMILES": Modality.SMILES,
            "amino_acid": Modality.AMINO_ACID,
            "nucleotide": Modality.NUCLEOTIDE,
            "3d_coordinates": Modality.COORDINATES_3D,
        }

        for modality_str, modality_enum in modality_map.items():
            tokenizer = ume.get_tokenizer(modality_str)

            assert tokenizer == mock_tokenizers[modality_enum]

    @patch("lobster.model._ume.FlexBERT.load_from_checkpoint")
    def test_get_embeddings_basic(self, mock_load_checkpoint, smiles_examples, protein_examples, dna_examples):
        """Test basic embedding functionality for all modalities"""
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
        ume = Ume.load_from_checkpoint("dummy_checkpoint.ckpt")

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

                # Verify tokenizer was called with the correct inputs
                mock_tokenizer.assert_called_with(
                    test_inputs[modality],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=mock_model.max_length,
                )

                # Test token-level embeddings
                token_embeddings = ume.get_embeddings(test_inputs[modality], modality, aggregate=False)
                assert token_embeddings.shape == (batch_size, seq_len, 768)
