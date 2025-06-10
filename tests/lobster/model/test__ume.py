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

    def test_configure_optimizers(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_flex_bert.return_value = mock_model

            with patch("lobster.model._ume.get_scheduler") as mock_get_scheduler:
                mock_get_scheduler.return_value = MagicMock()

                ume = Ume(
                    lr=1e-3,
                    beta1=0.9,
                    beta2=0.98,
                    eps=1e-12,
                    scheduler="constant_with_warmup",
                    num_training_steps=1000,
                    num_warmup_steps=100,
                )

                # Mock some dummy parameters so the optimizer doesn't get an empty list
                dummy_param = torch.nn.Parameter(torch.randn(10))
                with patch.object(ume, "parameters", return_value=[dummy_param]):
                    result = ume.configure_optimizers()

                    # Check that result has the expected structure
                    assert "optimizer" in result
                    assert "lr_scheduler" in result
                    assert isinstance(result["optimizer"], torch.optim.AdamW)

                    # Verify optimizer parameters
                    optimizer_params = list(result["optimizer"].param_groups[0]["params"])
                    assert len(optimizer_params) == 1
                    assert optimizer_params[0] is dummy_param

                    # Verify scheduler was called with correct parameters
                    mock_get_scheduler.assert_called_once_with(
                        "constant_with_warmup",
                        result["optimizer"],
                        num_training_steps=1000,
                        num_warmup_steps=100,
                        scheduler_specific_kwargs={},
                    )

    def test_train_mlm_step(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_flex_bert.return_value = mock_model

            ume = Ume()

            with patch.object(ume, "_mlm_step", return_value=torch.tensor(1.5)):
                batch = {
                    "input_ids": torch.randint(0, 100, (2, 1, 10)),
                    "attention_mask": torch.ones(2, 1, 10),
                    "modality": ["SMILES", "amino_acid"],
                }

                for step_method, step_name in [(ume.training_step, "train"), (ume.validation_step, "val")]:
                    loss = step_method(batch, 0)

                    assert isinstance(loss, torch.Tensor)
                    assert loss.item() == 1.5

                    ume._mlm_step.assert_called_with(batch, step_name)

                    ume._mlm_step.reset_mock()

    def test_contrastive_step_weight(self):
        with patch("lobster.model._ume.FlexBERT") as mock_flex_bert:
            mock_model = MagicMock()
            mock_flex_bert.return_value = mock_model

            # Initialize Ume with contrastive_loss_weight=0.5
            ume = Ume(contrastive_loss_weight=0.5)

            # Create mock objects for the methods
            mlm_mock = MagicMock(return_value=torch.tensor(2.0))
            infonce_mock = MagicMock(return_value=torch.tensor(4.0))
            split_batch_mock = MagicMock(return_value=({"input_ids": None}, {"input_ids": None}))

            # Replace the actual methods with mocks
            ume._mlm_step = mlm_mock
            ume._infonce_step = infonce_mock
            ume._split_combined_batch = split_batch_mock

            # Create a batch with two inputs (needed for contrastive learning)
            batch = {
                "input_ids": torch.randint(0, 100, (2, 2, 10)),  # 2 batch items, 2 inputs each
                "attention_mask": torch.ones(2, 2, 10),
                "modality": [["SMILES", "amino_acid"], ["amino_acid", "SMILES"]],
            }

            # Test training step
            loss = ume.training_step(batch, 0)

            # Expected combined loss: 0.5 * mlm_loss + 0.5 * contrastive_loss = 3.0
            expected_loss = 3.0

            assert isinstance(loss, torch.Tensor)
            assert loss.item() == expected_loss

            # Verify calls
            split_batch_mock.assert_called_once()
            mlm_mock.assert_called_once()
            infonce_mock.assert_called_once()

            # Test validation step
            split_batch_mock.reset_mock()
            mlm_mock.reset_mock()
            infonce_mock.reset_mock()

            loss = ume.validation_step(batch, 0)

            assert loss.item() == expected_loss
            split_batch_mock.assert_called_once()
            mlm_mock.assert_called_once()
            infonce_mock.assert_called_once()

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
            "3d_coordinates": [["aa", "bb", "cc", "dd"]],
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
            "3d_coordinates": [["aa", "bb", "cc", "dd"]],
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
            print(f"{embeddings_flash.dtype=}")
            print(f"{embeddings_no_flash.dtype=}")
            torch.testing.assert_close(
                embeddings_flash, embeddings_no_flash, rtol=1e-2, atol=1e-2
            )  # TODO investigate this tolerance

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

    def test_infonce_loss_disco_clip_equivalence(self):
        """Test that standard and DisCo-CLIP InfoNCE losses are equivalent in single-GPU scenario."""
        with (
            patch("lobster.model._ume.is_distributed", return_value=False),
            patch("lobster.model._ume.get_rank", return_value=0),
            patch("lobster.model._ume.Gather") as mock_gather,
        ):
            # Mock Gather to return the same embeddings (simulating single-GPU)
            mock_gather.side_effect = lambda x: x.clone()

            # Create one model - we'll test both loss methods on it
            ume = Ume(model_name="UME_mini", use_flash_attn=False)

            # Create simple test embeddings (32 samples as requested)
            batch_size = 32
            hidden_size = ume.embedding_dim

            torch.manual_seed(42)
            embeddings_a = torch.randn(batch_size, hidden_size)
            embeddings_b = torch.randn(batch_size, hidden_size)

            # Normalize (as done in actual implementation)
            embeddings_a = torch.nn.functional.normalize(embeddings_a, dim=-1)
            embeddings_b = torch.nn.functional.normalize(embeddings_b, dim=-1)

            # Test both loss methods on the same embeddings
            standard_loss = ume._standard_infonce_loss(embeddings_a, embeddings_b)
            disco_loss = ume._disco_infonce_loss(embeddings_a, embeddings_b)

            # Should be nearly identical (small floating-point differences expected)
            torch.testing.assert_close(standard_loss, disco_loss, rtol=1e-4, atol=1e-5)

            # Verify Gather was called twice for DisCo-CLIP method
            assert mock_gather.call_count == 2
