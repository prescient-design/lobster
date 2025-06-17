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
            mock_model._prepare_inputs.return_value = (
                torch.randint(0, 100, (1, 20)),
                torch.ones(1, 20),
                torch.tensor([0, 10, 20]),
            )
            mock_model.model.return_value = torch.randn(20, 768)
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
            mock_model._prepare_inputs.return_value = (
                torch.randint(0, 100, (1, 20)),
                torch.ones(1, 20),
                torch.tensor([0, 10, 20]),
            )
            mock_model.model.return_value = torch.randn(20, 768)
            mock_model.device = "cpu"
            mock_flex_bert.return_value = mock_model

            ume = Ume()

            # Test embed with aggregation
            inputs = {
                "input_ids": torch.randn(2, 1, 10),
                "attention_mask": torch.ones(2, 1, 10),
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

    def test_extract_batch_components(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume()

            # Create a sample batch with 2 views
            batch = {
                "input_ids": torch.randint(0, 100, (2, 2, 10)),  # (batch_size, num_views, seq_len)
                "attention_mask": torch.ones(2, 2, 10),
                "modality": [["SMILES", "amino_acid"], ["amino_acid", "SMILES"]],
            }

            # Test extracting first view
            view_0 = ume._extract_batch_components(batch, 0)
            assert view_0["input_ids"].shape == (2, 1, 10)
            assert view_0["attention_mask"].shape == (2, 1, 10)
            assert view_0["modality"] == ["SMILES", "amino_acid"]

            # Test extracting second view
            view_1 = ume._extract_batch_components(batch, 1)
            assert view_1["input_ids"].shape == (2, 1, 10)
            assert view_1["attention_mask"].shape == (2, 1, 10)
            assert view_1["modality"] == ["amino_acid", "SMILES"]

    def test_split_combined_batch(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume()

            # Create a sample batch with 2 views
            batch = {
                "input_ids": torch.randint(0, 100, (2, 2, 10)),
                "attention_mask": torch.ones(2, 2, 10),
                "modality": [["SMILES", "amino_acid"], ["amino_acid", "SMILES"]],
            }

            # Split the batch
            batches = ume._split_combined_batch(batch)
            assert len(batches) == 2

            # Check first view
            assert batches[0]["input_ids"].shape == (2, 1, 10)
            assert batches[0]["attention_mask"].shape == (2, 1, 10)
            assert batches[0]["modality"] == ["SMILES", "amino_acid"]

            # Check second view
            assert batches[1]["input_ids"].shape == (2, 1, 10)
            assert batches[1]["attention_mask"].shape == (2, 1, 10)
            assert batches[1]["modality"] == ["amino_acid", "SMILES"]

    def test_compute_loss_with_weighting(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = Ume(contrastive_loss_weight=0.5)

            # Create sample losses
            mlm_loss = torch.tensor(2.0)
            contrastive_loss = torch.tensor(4.0)

            # Test loss computation
            total_loss = ume._compute_loss_with_weighting(mlm_loss, contrastive_loss, "train")
            expected_loss = 0.5 * mlm_loss + 0.5 * contrastive_loss
            assert torch.allclose(total_loss, expected_loss)

    def test_delegate_step_by_batch_shape(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            # Test MLM only mode
            ume_mlm = Ume(contrastive_loss_type=None)
            batch_mlm = {
                "input_ids": torch.randint(0, 100, (2, 1, 10)),
                "attention_mask": torch.ones(2, 1, 10),
                "modality": ["SMILES", "amino_acid"],
            }

            with patch.object(ume_mlm, "_compute_mlm_loss", return_value=torch.tensor(1.0)):
                loss = ume_mlm._delegate_step_by_batch_shape(batch_mlm, "train")
                assert loss.item() == 1.0

            # Test InfoNCE mode
            ume_infonce = Ume(contrastive_loss_type="clip")
            batch_infonce = {
                "input_ids": torch.randint(0, 100, (2, 2, 10)),
                "attention_mask": torch.ones(2, 2, 10),
                "modality": [["SMILES", "amino_acid"], ["amino_acid", "SMILES"]],
            }

            with patch.object(ume_infonce, "_infonce_step", return_value=torch.tensor(2.0)):
                loss = ume_infonce._delegate_step_by_batch_shape(batch_infonce, "train")
                assert loss.item() == 2.0

            # Test Symile mode
            ume_symile = Ume(contrastive_loss_type="symile")
            batch_symile = {
                "input_ids": torch.randint(0, 100, (2, 3, 10)),  # 3 views for Symile
                "attention_mask": torch.ones(2, 3, 10),
                "modality": [["SMILES", "amino_acid", "nucleotide"], ["amino_acid", "SMILES", "nucleotide"]],
            }

            with patch.object(ume_symile, "_symile_step", return_value=torch.tensor(3.0)):
                loss = ume_symile._delegate_step_by_batch_shape(batch_symile, "train")
                assert loss.item() == 3.0

    def test_embed_sequences_cpu(self):
        """Test Ume's embed_sequences method without flash-attn on CPU."""
        # Initialize Ume with a small model and flash-attn disabled
        ume = Ume(
            model_name="UME_mini",
            max_length=10,
            use_flash_attn=False,  # Disable flash-attn
        )

        # Test sequences for each modality
        test_sequences = {
            "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
            "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL"],
            "nucleotide": ["ATGCATGC"],
        }

        # Test embedding for each modality
        for modality, sequences in test_sequences.items():
            # Get embeddings without aggregation
            embeddings = ume.embed_sequences(sequences, modality, aggregate=False)
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.dim() == 3  # [batch_size, seq_length, hidden_size]
            assert embeddings.shape[0] == len(sequences)
            assert embeddings.shape[1] <= ume.max_length

            # Get embeddings with aggregation
            embeddings = ume.embed_sequences(sequences, modality, aggregate=True)
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.dim() == 2  # [batch_size, hidden_size]
            assert embeddings.shape[0] == len(sequences)

    def test_embed_sequences_gpu_flash_attn(self):
        """Test Ume's embed_sequences method with and without flash-attn on GPU."""
        # Skip if not on GPU
        if not torch.cuda.is_available():
            pytest.skip("This test requires a GPU")

        # Test sequences for each modality
        test_sequences = {
            "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
            "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL"],
            "nucleotide": ["ATGCATGC"],
        }

        # Initialize Ume with flash-attn enabled
        ume_flash = Ume(
            model_name="UME_mini",
            max_length=10,
            use_flash_attn=True,
        )
        ume_flash = ume_flash.cuda()
        ume_flash.eval()

        # Initialize Ume with flash-attn disabled
        ume_no_flash = Ume(
            model_name="UME_mini",
            max_length=10,
            use_flash_attn=False,
        )
        ume_no_flash = ume_no_flash.cuda()
        ume_no_flash.load_state_dict(ume_flash.state_dict())
        ume_no_flash.eval()

        # Test embedding for each modality
        for modality, sequences in test_sequences.items():
            # Get embeddings with flash-attn
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            embeddings_flash = ume_flash.embed_sequences(sequences, modality, aggregate=False)
            end_time.record()
            torch.cuda.synchronize()
            flash_time = start_time.elapsed_time(end_time)

            # Get embeddings without flash-attn
            start_time.record()
            embeddings_no_flash = ume_no_flash.embed_sequences(sequences, modality, aggregate=False)
            end_time.record()
            torch.cuda.synchronize()
            no_flash_time = start_time.elapsed_time(end_time)

            # Verify shapes and values
            assert embeddings_flash.shape == embeddings_no_flash.shape
            torch.testing.assert_close(embeddings_flash, embeddings_no_flash, rtol=1e-2, atol=1e-2)

            # Log performance difference
            speedup = no_flash_time / flash_time
            diff = embeddings_flash - embeddings_no_flash
            print(f"\nModality: {modality}")
            print(f"Flash-attn time: {flash_time:.2f}ms")
            print(f"No flash-attn time: {no_flash_time:.2f}ms")
            print(f"Speedup: {speedup:.2f}x")
            print(f"{diff.abs().max()=}")
            print(f"{diff.abs().mean()=}")

            # Test with aggregation
            embeddings_flash_agg = ume_flash.embed_sequences(sequences, modality, aggregate=True)
            embeddings_no_flash_agg = ume_no_flash.embed_sequences(sequences, modality, aggregate=True)

            assert embeddings_flash_agg.shape == embeddings_no_flash_agg.shape
