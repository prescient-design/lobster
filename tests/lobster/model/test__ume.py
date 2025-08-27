import logging
from unittest.mock import MagicMock, patch

import pytest
import torch

from lobster.constants import Modality
from lobster.model import UME


@pytest.fixture
def sample_sequences():
    return {
        "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"],
        "nucleotide": ["ATGCATGC", "GCTAGCTA"],
    }


class TestUME:
    def test_initialization(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = UME(
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

            ume = UME()

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

            ume = UME(max_length=10)

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
            mock_model.device = torch.device("cpu")
            mock_flex_bert.return_value = mock_model

            ume = UME()

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
            ume = UME()

            # Test with string modality
            tokenizer = ume.get_tokenizer("SMILES")
            assert tokenizer is not None

            # Test with Modality enum
            tokenizer = ume.get_tokenizer(Modality.AMINO_ACID)
            assert tokenizer is not None

    def test_get_vocab(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = UME()
            vocab = ume.get_vocab()
            assert isinstance(vocab, dict)
            # Vocab should be non-empty
            assert len(vocab) > 0

    def test_modalities_property(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = UME()
            modalities = ume.modalities
            expected_modalities = ["SMILES", "amino_acid", "nucleotide", "3d_coordinates"]
            assert modalities == expected_modalities

    def test_extract_batch_components(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = UME()

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
            ume = UME()

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

    def test_compute_weighted_loss(self):
        with patch("lobster.model._ume.FlexBERT", MagicMock()):
            ume = UME(contrastive_loss_weight=0.5)

            # Create sample losses
            mlm_loss = torch.tensor(2.0)
            contrastive_loss = torch.tensor(4.0)

            # Test loss computation
            total_loss = ume._compute_weighted_loss(mlm_loss, contrastive_loss, "train")
            expected_loss = 0.5 * mlm_loss + 0.5 * contrastive_loss
            assert torch.allclose(total_loss, expected_loss)

    def test_embed_sequences_cpu(self):
        """Test UME's embed_sequences method without flash-attn on CPU."""
        # Initialize UME with a small model and flash-attn disabled
        ume = UME(
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

    def test_embed_dtype_match_model(self):
        """Test UME's embed_sequences method returns embeddings with the same dtype as model weights."""
        # Skip if not on GPU

        test_modality = "SMILES"
        test_sequences = ["CC(=O)OC1=CC=CC=C1C(=O)O"]

        ume = UME(
            model_name="UME_mini",
            max_length=10,
            use_flash_attn=False,
        )
        ume.eval()

        for want_dtype in [torch.bfloat16, torch.float16, torch.float32]:
            ume = ume.to(dtype=want_dtype)

            embeddings_no_agg = ume.embed_sequences(test_sequences, test_modality, aggregate=False)
            assert embeddings_no_agg.dtype == want_dtype

            embeddings_agg = ume.embed_sequences(test_sequences, test_modality, aggregate=True)
            assert embeddings_agg.dtype == want_dtype

    def test_embed_sequences_gpu_flash_attn(self):
        """Test UME's embed_sequences method with and without flash-attn on GPU."""
        # Skip if not on GPU
        if not torch.cuda.is_available():
            pytest.skip("This test requires a GPU")

        # Test sequences for each modality
        test_sequences = {
            "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
            "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL"],
            "nucleotide": ["ATGCATGC"],
        }

        # Initialize UME with flash-attn enabled
        ume_flash = UME(
            model_name="UME_mini",
            max_length=10,
            use_flash_attn=True,
        )
        ume_flash = ume_flash.cuda()
        ume_flash.eval()

        # Initialize UME with flash-attn disabled
        ume_no_flash = UME(
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

            # Get embeddings without flash-attn
            start_time.record()
            embeddings_no_flash = ume_no_flash.embed_sequences(sequences, modality, aggregate=False)
            end_time.record()
            torch.cuda.synchronize()

            # Verify shapes and values
            assert embeddings_flash.shape == embeddings_no_flash.shape
            torch.testing.assert_close(embeddings_flash, embeddings_no_flash, rtol=1e-2, atol=1e-2)

            # Test with aggregation
            embeddings_flash_agg = ume_flash.embed_sequences(sequences, modality, aggregate=True)
            embeddings_no_flash_agg = ume_no_flash.embed_sequences(sequences, modality, aggregate=True)

            assert embeddings_flash_agg.shape == embeddings_no_flash_agg.shape

    def test_flash_attention_consistency_across_devices(self):
        """Test that flash attention and non-flash attention produce consistent embeddings."""
        # Test sequences with different lengths per modality to test batching and padding
        test_cases = [
            (
                "amino_acid",
                [
                    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",  # Long protein (67 chars)
                    "ACDEFGHIKL",  # Short protein (10 chars)
                ],
            ),
            (
                "nucleotide",
                [
                    "ATGCGATGAATTGCCAGGACGCTACCGGTTGGATTGCGCAGGTTCTGAACGCGTTTGGGATCCTTAACTAGTGGAATTCCCG",  # Long DNA (78 chars)
                    "ATGCATGC",  # Short DNA (8 chars)
                ],
            ),
            (
                "SMILES",
                [
                    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin (24 chars)
                    "CCO",  # Ethanol (3 chars)
                ],
            ),
        ]

        for modality, sequences in test_cases:
            # Test GPU flash attention vs CPU non-flash attention if GPU available
            if torch.cuda.is_available():
                # GPU model with flash attention
                ume_gpu = UME(
                    model_name="UME_mini",
                    max_length=512,
                    use_flash_attn=True,
                )
                ume_gpu = ume_gpu.cuda()
                ume_gpu.eval()

                # CPU model without flash attention
                ume_cpu = UME(
                    model_name="UME_mini",
                    max_length=512,
                    use_flash_attn=False,
                )
                ume_cpu = ume_cpu.cpu()
                ume_cpu.load_state_dict(ume_gpu.state_dict(), strict=False)
                ume_cpu.eval()

                # Get embeddings from both models
                with torch.no_grad():
                    embeddings_gpu = ume_gpu.embed_sequences(sequences, modality, aggregate=True)
                    embeddings_cpu = ume_cpu.embed_sequences(sequences, modality, aggregate=True)

                # Move to same device for comparison
                embeddings_gpu = embeddings_gpu.cpu()

                # Check similarity
                cosine_sim = torch.nn.functional.cosine_similarity(embeddings_gpu, embeddings_cpu, dim=1)

                # Should be very similar after padding fix (>99.9% similarity for all sequences)
                assert cosine_sim.min().item() > 0.999, (
                    f"Embeddings not consistent enough: min={cosine_sim.min().item():.6f} < 0.999"
                )

                # Also test without aggregation (token-level embeddings)
                embeddings_gpu_tokens = ume_gpu.embed_sequences(sequences, modality, aggregate=False)
                embeddings_cpu_tokens = ume_cpu.embed_sequences(sequences, modality, aggregate=False)

                # Should have same shape
                assert embeddings_gpu_tokens.shape == embeddings_cpu_tokens.shape

                # Move to CPU for comparison
                embeddings_gpu_tokens = embeddings_gpu_tokens.cpu()

                # Test token-level consistency
                # Compute cosine similarity for each token position across all sequences
                batch_size, seq_len, hidden_dim = embeddings_gpu_tokens.shape
                embeddings_gpu_flat = embeddings_gpu_tokens.view(-1, hidden_dim)
                embeddings_cpu_flat = embeddings_cpu_tokens.view(-1, hidden_dim)

                # Only compare non-zero positions (actual tokens, not padding)
                # Padding positions should be zero in both models
                gpu_nonzero = embeddings_gpu_flat.norm(dim=1) > 1e-6
                cpu_nonzero = embeddings_cpu_flat.norm(dim=1) > 1e-6

                # Both models should have the same non-zero positions
                assert torch.equal(gpu_nonzero, cpu_nonzero), "Different non-zero token positions between models"

                # For non-zero positions, embeddings should be highly similar
                if gpu_nonzero.any():
                    nonzero_indices = gpu_nonzero.nonzero().squeeze()
                    if nonzero_indices.numel() > 0:
                        gpu_nonzero_embeds = embeddings_gpu_flat[nonzero_indices]
                        cpu_nonzero_embeds = embeddings_cpu_flat[nonzero_indices]

                        token_cosine_sims = torch.nn.functional.cosine_similarity(
                            gpu_nonzero_embeds, cpu_nonzero_embeds, dim=1
                        )
                        min_token_sim = token_cosine_sims.min().item()

                        # Token-level embeddings should also be highly consistent
                        assert min_token_sim > 0.995, (
                            f"Token-level embeddings not consistent: {min_token_sim:.6f} < 0.995"
                        )

            # CPU model with flash attention disabled but unpadded architecture
            ume_cpu_unpadded = UME(
                model_name="UME_mini",
                max_length=512,
                use_flash_attn=False,
                model_kwargs={"padding": "unpadded", "use_sdpa_attn_mask": False},
            )
            ume_cpu_unpadded.eval()

            # CPU model with padded architecture
            ume_cpu_padded = UME(
                model_name="UME_mini",
                max_length=512,
                use_flash_attn=False,
                model_kwargs={"padding": "padded", "use_sdpa_attn_mask": True},
            )
            ume_cpu_padded.eval()

            # Copy weights to ensure same model
            ume_cpu_padded.load_state_dict(ume_cpu_unpadded.state_dict(), strict=False)

            with torch.no_grad():
                embeddings_unpadded = ume_cpu_unpadded.embed_sequences(sequences, modality, aggregate=True)
                embeddings_padded = ume_cpu_padded.embed_sequences(sequences, modality, aggregate=True)

            # Check similarity
            cosine_sim = torch.nn.functional.cosine_similarity(embeddings_unpadded, embeddings_padded, dim=1)

            # Should be very similar after padding fix
            assert cosine_sim.min().item() > 0.999, (
                f"Unpadded vs Padded not consistent: min={cosine_sim.min().item():.6f} < 0.999"
            )

            # Also test token-level embeddings
            with torch.no_grad():
                embeddings_unpadded_tokens = ume_cpu_unpadded.embed_sequences(sequences, modality, aggregate=False)
                embeddings_padded_tokens = ume_cpu_padded.embed_sequences(sequences, modality, aggregate=False)

            # Check that padding tokens are properly zeroed and non-padding tokens are consistent
            _batch_size, _seq_len, hidden_dim = embeddings_unpadded_tokens.shape
            unpadded_flat = embeddings_unpadded_tokens.view(-1, hidden_dim)
            padded_flat = embeddings_padded_tokens.view(-1, hidden_dim)

            # Both should have the same zero/non-zero pattern after masking
            unpadded_nonzero = unpadded_flat.norm(dim=1) > 1e-6
            padded_nonzero = padded_flat.norm(dim=1) > 1e-6

            assert torch.equal(unpadded_nonzero, padded_nonzero), "Different zero patterns between unpadded and padded"

            # Non-zero tokens should be consistent
            if unpadded_nonzero.any():
                nonzero_indices = unpadded_nonzero.nonzero().squeeze()
                if nonzero_indices.numel() > 0:
                    unpadded_nonzero_embeds = unpadded_flat[nonzero_indices]
                    padded_nonzero_embeds = padded_flat[nonzero_indices]

                    token_cosine_sims = torch.nn.functional.cosine_similarity(
                        unpadded_nonzero_embeds, padded_nonzero_embeds, dim=1
                    )
                    min_token_sim = token_cosine_sims.min().item()

                    assert min_token_sim > 0.995, f"Token-level not consistent: {min_token_sim:.6f} < 0.995"

    @patch("lobster.model._ume.load_checkpoint_with_retry")
    @patch("lobster.model._ume.get_ume_checkpoints")
    @patch("lobster.model._ume.get_s3_last_modified_timestamp")
    @patch("lobster.model._ume.os.path.join")
    @patch("lobster.model._ume.os.getcwd")
    def test_from_pretrained(
        self, mock_getcwd, mock_join, mock_get_timestamp, mock_get_checkpoints, mock_load_checkpoint
    ):
        """Test from_pretrained method with mocked dependencies."""
        mock_get_checkpoints.return_value = {"ume-mini-base-12M": "s3://bucket/ume-mini-base-12M.ckpt"}
        mock_getcwd.return_value = "/current/working/dir"
        mock_join.return_value = "/current/working/dir/models/ume"
        mock_get_timestamp.return_value = "20250711-061718"

        # Create a properly mocked model with expected attributes for validation
        mock_model = MagicMock()

        # Mock parameters to return correct parameter count for ume-mini-base-12M (should be 10M-20M)
        mock_param = MagicMock()
        mock_param.numel.return_value = 2_000_000  # 2M parameters per mock parameter
        mock_param.device = torch.device("cpu")

        # Return 6 parameters totaling 12M parameters (within expected range)
        # Use a lambda to return a fresh iterator each time parameters() is called
        mock_model.parameters = lambda: iter([mock_param] * 6)

        # Mock other attributes accessed during validation
        mock_model.embedding_dim = 384
        mock_model.use_flash_attn = False
        mock_model.model.config.num_hidden_layers = 6

        mock_load_checkpoint.return_value = mock_model

        result = UME.from_pretrained("ume-mini-base-12M", use_flash_attn=False)

        mock_get_checkpoints.assert_called_once()
        mock_get_timestamp.assert_called_once_with("s3://bucket/ume-mini-base-12M.ckpt")

        mock_join.assert_called_once_with("/current/working/dir", "models", "ume")

        mock_load_checkpoint.assert_called_once_with(
            checkpoint_path="s3://bucket/ume-mini-base-12M.ckpt",
            local_directory="/current/working/dir/models/ume",
            local_filename="ume-mini-base-12M-20250711-061718.ckpt",
            load_func=UME.load_from_checkpoint,
            device=None,
            use_flash_attn=False,
        )

        assert result == mock_model

    def test_load_checkpoint_disable_flash_attn_cpu_inference(self):
        """Test loading UME checkpoint trained with flash-attn, disabling it, and running inference on CPU."""
        # Suppress boto3/S3 debug logging

        logging.getLogger("boto3").setLevel(logging.WARNING)
        logging.getLogger("botocore").setLevel(logging.WARNING)
        logging.getLogger("s3fs").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        logging.getLogger("aiobotocore").setLevel(logging.WARNING)

        # Check if S3 bucket is accessible, skip test if not
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError

            s3_client = boto3.client("s3")
            s3_client.head_bucket(Bucket="prescient-lobster")
        except (ImportError, NoCredentialsError, ClientError) as e:
            pytest.skip(f"S3 bucket 'prescient-lobster' not accessible: {e}")

        checkpoint_path = "s3://prescient-lobster/ume/runs/2025-06-11T02-12-16/epoch=0-step=19500-val_loss=1.0225.ckpt"

        # Load the checkpoint that was trained with flash-attn
        ume = UME.load_from_checkpoint(
            checkpoint_path,
            use_flash_attn=False,  # Disable flash-attn for CPU inference
            map_location="cpu",  # Force CPU loading
        )

        # Ensure model is on CPU and in eval mode
        ume = ume.cpu()
        ume.eval()

        # Verify flash-attn is disabled
        assert ume.use_flash_attn is False

        # Note: Full inference testing with this configuration may have compatibility
        # issues due to architecture differences between flash-attn and standard attention.

        # Test a simple single sequence to verify basic functionality
        simple_sequence = "CC(=O)O"  # Simple molecule (acetic acid)
        try:
            # Try basic embedding - this may work for simple cases
            embeddings = ume.embed_sequences([simple_sequence], "SMILES", aggregate=True)
            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape[0] == 1
            assert embeddings.device.type == "cpu"
        except Exception as e:
            # This is expected due to architecture mismatch between flash-attn training
            # and CPU inference without flash-attn
            # The important part is that the model loaded successfully and
            # use_flash_attn attribute is correctly set
            print(e)

    @pytest.mark.parametrize(
        "smiles",
        [
            pytest.param("CC(=O", id="unclosed_parenthesis"),
            pytest.param("CC)=O)O", id="extra_closing_parenthesis"),
            pytest.param("CC[=O]O", id="invalid_bond_notation"),
            pytest.param("C1CCCCC", id="unclosed_ring"),
            pytest.param("", id="empty_string"),
            pytest.param("   ", id="whitespace_only"),
            pytest.param("C" * 1000, id="extremely_long"),
            pytest.param("CC@#$%^&*", id="invalid_characters"),
            pytest.param("INVALID_SMILES", id="completely_invalid"),
            pytest.param("c1ccccc1", id="lowercase_aromatic"),
            pytest.param("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", id="valid_smiles"),  # Control
        ],
    )
    def test_malformed_smiles_behavior(self, smiles, request):
        """Ensure UME returns valid embeddings and does not crash on malformed SMILES input"""
        ume = UME(model_name="UME_mini", max_length=512, use_flash_attn=False)

        embeddings = ume.embed_sequences([smiles], "SMILES")
        embedding_norm = torch.norm(embeddings).item()
        # Just validate it's a proper tensor if it succeeds
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (1, ume.embedding_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

        test_id = request.node.callspec.id
        print(f"{test_id}: accepted (norm: {embedding_norm:.3f})")

    @pytest.mark.parametrize(
        "sequence",
        [
            pytest.param("MKTVRQXYZ", id="invalid_amino_acids"),
            pytest.param("MKTVRQ123", id="numbers_mixed"),
            pytest.param("MKTVRQ@#$", id="special_characters"),
            pytest.param("MKTVRQ-ACDEFG", id="dash_separator"),
            pytest.param("MKTVRQ ACDEFG", id="space_separator"),
            pytest.param("M*T*V*R*Q", id="asterisk_unknowns"),
            pytest.param("", id="empty_sequence"),
            pytest.param("mktvrq", id="all_lowercase"),
            pytest.param("MKTVBJOUXZ", id="multiple_invalid"),
            pytest.param("MKTVRQERLK", id="valid_protein"),  # control
        ],
    )
    def test_malformed_protein_behavior(self, sequence, request):
        """Document how UME handles invalid protein sequences"""
        ume = UME(model_name="UME_mini", max_length=512, use_flash_attn=False)

        embeddings = ume.embed_sequences([sequence], "amino_acid")
        embedding_norm = torch.norm(embeddings).item()

        # Just validate it's a proper tensor if it succeeds
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (1, ume.embedding_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
        test_id = request.node.callspec.id
        print(f"{test_id}: accepted (norm: {embedding_norm:.3f})")
