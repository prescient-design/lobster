from unittest.mock import MagicMock, patch

import pytest
import torch
from lobster.constants import Modality
from lobster.model import Ume


@pytest.fixture
def sample_sequences():
    return {
        "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"],
        "nucleotide": ["ATGCATGC", "GCTAGCTA"],
        "3d_coordinates": [["aa", "bb", "cc", "dd"], ["aa", "bb", "cc", "dd"]],
    }


class TestUme:
    def test_initialization(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume(
                model_name="UME_mini",
                max_length=10,
                scheduler="warmup_stable_decay",
                num_training_steps=100,
                num_warmup_steps=10,
                scheduler_kwargs={"num_decay_steps": 10},
            )
            assert ume.model is not None
            assert ume.frozen is False
            assert ume.max_length == 10

    def test_freeze_unfreeze(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_param = MagicMock()
            mock_param.requires_grad = True
            mock_model.parameters.return_value = [mock_param]
            mock_flex_bert.return_value = mock_model

            ume = Ume()

            # Test freeze
            ume.freeze()
            assert ume.frozen is True
            assert not mock_param.requires_grad
            mock_model.eval.assert_called_once()

            # Test unfreeze
            ume.unfreeze()
            assert ume.frozen is False
            assert mock_param.requires_grad
            mock_model.train.assert_called_once()

    def test_embed_sequences(self, sample_sequences):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.tokens_to_latents.return_value = torch.randn(2, 10, 768)
            mock_model.device = "cpu"
            mock_flex_bert.return_value = mock_model

            ume = Ume(max_length=10)

            # Test each modality
            for modality, sequences in sample_sequences.items():
                embeddings = ume.embed_sequences(sequences, modality, aggregate=False)
                assert embeddings.shape == (2, 10, 768)

    def test_embed(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.tokens_to_latents.return_value = torch.randn(2, 10, 768)
            mock_model.device = "cpu"
            mock_flex_bert.return_value = mock_model

            ume = Ume()

            # Test embed with aggregation
            inputs = {
                "input_ids": torch.randn(2, 10),
                "attention_mask": torch.ones(2, 10),
            }
            embeddings = ume.embed(inputs, aggregate=True)
            assert embeddings.shape == (2, 768)

            # Test embed without aggregation
            embeddings = ume.embed(inputs, aggregate=False)
            assert embeddings.shape == (2, 10, 768)

    def test_get_tokenizer(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume()

            # Test with string modality
            tokenizer = ume.get_tokenizer("SMILES")
            assert tokenizer is not None

            # Test with Modality enum
            tokenizer = ume.get_tokenizer(Modality.AMINO_ACID)
            assert tokenizer is not None

    def test_get_vocab(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume()
            vocab = ume.get_vocab()
            assert isinstance(vocab, dict)
            # Vocab should be non-empty
            assert len(vocab) > 0

    def test_modalities_property(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume()
            modalities = ume.modalities
            expected_modalities = ["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]
            assert modalities == expected_modalities

    def test_configure_optimizers(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_model.configure_optimizers.return_value = {"optimizer": "mock_optimizer"}
            mock_flex_bert.return_value = mock_model

            ume = Ume()
            result = ume.configure_optimizers()
            assert result == {"optimizer": "mock_optimizer"}
            mock_model.configure_optimizers.assert_called_once()

    def test_training_step(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_model._compute_loss.return_value = (torch.tensor(1.0), torch.tensor([1.0, 2.0]))
            mock_flex_bert.return_value = mock_model

            ume = Ume()
            batch = {
                "input_ids": torch.randn(2, 10),
                "attention_mask": torch.ones(2, 10),
                "modality": ["SMILES", "SMILES"],
            }
            loss = ume.training_step(batch, 0)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() == 1.0

    def test_validation_step(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_model._compute_loss.return_value = (torch.tensor(1.5), torch.tensor([1.5, 1.5]))
            mock_flex_bert.return_value = mock_model

            ume = Ume()
            batch = {
                "metadata": {"modality": ["SMILES", "SMILES"]},
                "input_ids": torch.randn(2, 10),
                "attention_mask": torch.ones(2, 10),
            }
            loss = ume.validation_step(batch, 0)
            assert isinstance(loss, torch.Tensor)
            assert loss.item() == 1.5
