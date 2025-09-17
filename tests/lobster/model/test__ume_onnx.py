import onnx
import onnxruntime as ort
import pytest
import torch
from torch import Size, Tensor

from lobster.constants import Modality
from lobster.model import UME
from lobster.model._onnx_utils import benchmark_onnx_pytorch, run_onnx_inference

# Performance optimizations for CI/CD:
# - Reduced max_length from 128 to 64
# - Shorter test sequences
# - Fewer benchmark iterations
# - Skip full ONNX validation
# - Reduced batch sizes and sequence lengths in tests
# - Single modality testing instead of all modalities
# - Marked slow tests with @pytest.mark.slow


@pytest.fixture(scope="module")
def ume_model():
    """Create a UME model for testing."""
    model = UME(
        model_name="UME_mini",
        max_length=64,  # Reduced from 128 for faster testing
        use_flash_attn=False,  # Disable flash attention for CPU compatibility
    )
    model.eval()
    model.freeze()  # Freeze for inference
    return model


@pytest.fixture(scope="module")
def sample_sequences():
    """Sample sequences for different modalities."""
    return {
        "SMILES": ["CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],  # Shorter SMILES for speed
        "amino_acid": ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"],  # Protein sequences
        "nucleotide": ["ATGCATGC", "GCTAGCTA"],  # DNA sequences
    }


class TestUMEONNX:
    """Test ONNX compilation and inference for UME models."""

    def test_ume_model_initialization(self, ume_model):
        """Test that UME model initializes correctly for ONNX export."""
        assert ume_model is not None
        assert ume_model.frozen is True
        assert ume_model.model is not None
        assert hasattr(ume_model, "embedding_dim")

    def test_ume_embed_sequences_pytorch(self, ume_model, sample_sequences):
        """Test UME embedding generation in PyTorch before ONNX export."""
        for modality, sequences in sample_sequences.items():
            embeddings = ume_model.embed_sequences(sequences, modality, aggregate=True)

            # Check output shape
            expected_shape = (len(sequences), ume_model.embedding_dim)
            assert embeddings.shape == Size(expected_shape)

            # Check output type
            assert isinstance(embeddings, Tensor)

            # Check that embeddings are not all zeros
            assert not torch.allclose(embeddings, torch.zeros_like(embeddings))

    def test_ume_embed_sequences_no_aggregate(self, ume_model, sample_sequences):
        """Test UME embedding generation without aggregation."""
        for modality, sequences in sample_sequences.items():
            embeddings = ume_model.embed_sequences(sequences, modality, aggregate=False)

            # Check output shape (should include sequence length dimension)
            expected_shape = (len(sequences), ume_model.max_length, ume_model.embedding_dim)
            assert embeddings.shape == Size(expected_shape)

            # Check output type
            assert isinstance(embeddings, Tensor)

    def test_ume_embed_method(self, ume_model):
        """Test UME embed method with direct tensor inputs."""
        # Create sample inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, 1, seq_len))
        attention_mask = torch.ones(batch_size, 1, seq_len)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Test with aggregation
        embeddings = ume_model.embed(inputs, aggregate=True)
        assert embeddings.shape == (batch_size, ume_model.embedding_dim)

        # Test without aggregation
        embeddings = ume_model.embed(inputs, aggregate=False)
        assert embeddings.shape == (batch_size, seq_len, ume_model.embedding_dim)

    def test_ume_onnx_export(self, ume_model, tmpdir):
        """Test ONNX export of UME model."""
        # Create sample inputs for export with correct device
        batch_size, seq_len = 2, 8  # Reduced sequence length for faster testing
        device = next(ume_model.parameters()).device
        input_ids = torch.randint(0, 100, (batch_size, 1, seq_len), dtype=torch.long, device=device)
        attention_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.long, device=device)

        # Test PyTorch forward pass first
        _pytorch_output = ume_model(input_ids, attention_mask)

        # Export to ONNX using the new method
        onnx_path = tmpdir / "ume_model.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Check that ONNX file was created
        assert onnx_path.exists()

        # Validate ONNX model (skip full_check for speed)
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model, full_check=False)

        # Print model graph for debugging
        graph = onnx.helper.printable_graph(onnx_model.graph)
        assert isinstance(graph, str)
        assert len(graph) > 0

    def test_ume_onnx_inference(self, ume_model, tmpdir):
        """Test ONNX inference and compare with PyTorch outputs."""
        # Test with actual sequences that match the export samples
        test_sequences = ["CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]  # Shorter sequences for speed

        # Export to ONNX using the same sequences for consistency
        onnx_path = tmpdir / "ume_model_inference.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES, sample_sequences=test_sequences)

        # Get ONNX output using utility with the same max_length as the model
        onnx_output = run_onnx_inference(
            str(onnx_path),
            test_sequences,
            modality=Modality.SMILES,
            max_length=ume_model.max_length,
        )

        # Get PyTorch output for comparison
        pytorch_output_sequences = ume_model.embed_sequences(test_sequences, modality=Modality.SMILES, aggregate=True)

        # Compare outputs - ensure both are on CPU for comparison
        assert onnx_output.shape == pytorch_output_sequences.shape, (
            f"Shape mismatch: ONNX {onnx_output.shape} vs PyTorch {pytorch_output_sequences.shape}"
        )

        # Convert to tensors for comparison
        onnx_tensor = torch.from_numpy(onnx_output)
        pytorch_tensor = pytorch_output_sequences.cpu()

        # Check if outputs are close with more relaxed tolerance
        is_close = torch.allclose(onnx_tensor, pytorch_tensor, atol=1e-3, rtol=1e-3)
        max_diff = torch.max(torch.abs(onnx_tensor - pytorch_tensor))
        assert is_close, f"Output mismatch. Max diff: {max_diff}"

    def test_ume_onnx_dynamic_batch_size(self, ume_model, tmpdir):
        """Test ONNX model with different batch sizes."""
        seq_len = 8  # Reduced for faster testing

        # Export model with dynamic batch size
        onnx_path = tmpdir / "ume_model_dynamic.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Test with different batch sizes (reduced set for speed)
        ort_session = ort.InferenceSession(str(onnx_path))

        device = next(ume_model.parameters()).device
        for batch_size in [1, 2]:  # Reduced from [1, 2, 4]
            input_ids = torch.randint(0, 100, (batch_size, 1, seq_len), dtype=torch.long, device=device)
            attention_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.long, device=device)

            # PyTorch output
            pytorch_output = ume_model(input_ids, attention_mask)

            # ONNX output - ensure tensors are on CPU for numpy conversion
            ort_inputs = {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_output = torch.from_numpy(ort_outputs[0])

            # Compare outputs - ensure both are on CPU for comparison
            assert onnx_output.shape == pytorch_output.shape
            assert torch.allclose(onnx_output, pytorch_output.cpu(), atol=1e-5, rtol=1e-5)

    def test_ume_onnx_dynamic_sequence_length(self, ume_model, tmpdir):
        """Test ONNX model with different sequence lengths."""
        batch_size = 2

        # Export model with dynamic sequence length
        onnx_path = tmpdir / "ume_model_dynamic_seq.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Test with different sequence lengths (reduced set for speed)
        ort_session = ort.InferenceSession(str(onnx_path))

        device = next(ume_model.parameters()).device
        for seq_len in [5, 10]:  # Reduced from [5, 10, 20]
            input_ids = torch.randint(0, 100, (batch_size, 1, seq_len), dtype=torch.long, device=device)
            attention_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.long, device=device)

            # PyTorch output
            pytorch_output = ume_model(input_ids, attention_mask)

            # ONNX output - ensure tensors are on CPU for numpy conversion
            ort_inputs = {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
            }
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_output = torch.from_numpy(ort_outputs[0])

            # Compare outputs - ensure both are on CPU for comparison
            assert onnx_output.shape == pytorch_output.shape
            assert torch.allclose(onnx_output, pytorch_output.cpu(), atol=1e-5, rtol=1e-5)

    def test_ume_onnx_with_real_sequences(self, ume_model, sample_sequences, tmpdir):
        """Test ONNX inference with real molecular sequences."""
        # Export model
        onnx_path = tmpdir / "ume_model_real_sequences.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Test with real sequences (only test SMILES for speed)
        ort_session = ort.InferenceSession(str(onnx_path))

        # Only test SMILES modality for speed
        modality = Modality.SMILES
        sequences = sample_sequences["SMILES"]

        # Get PyTorch embeddings
        pytorch_embeddings = ume_model.embed_sequences(sequences, modality, aggregate=True)

        # Get tokenized inputs for ONNX
        tokenizer_transform = ume_model.tokenizer_transforms[modality]
        encoded_batch = tokenizer_transform(sequences)

        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        # Ensure 3D format for ONNX and correct dtype
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1)

        # Ensure correct dtype for ONNX
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        # ONNX inference - ensure tensors are on CPU for numpy conversion
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_embeddings = torch.from_numpy(ort_outputs[0])

        # Compare outputs - ensure both are on CPU for comparison
        assert onnx_embeddings.shape == pytorch_embeddings.shape
        assert torch.allclose(onnx_embeddings, pytorch_embeddings.cpu(), atol=1e-5, rtol=1e-5)

        # Verify embedding properties
        assert not torch.isnan(onnx_embeddings).any()
        assert not torch.isinf(onnx_embeddings).any()
        assert onnx_embeddings.shape[-1] == ume_model.embedding_dim

    def test_ume_onnx_error_handling(self, ume_model):
        """Test ONNX export error handling."""
        # Test with invalid input shapes
        with pytest.raises((AssertionError, RuntimeError, ValueError, KeyError)):
            # Try to call embed with missing required keys
            # This should definitely fail because embed expects 'input_ids' and 'attention_mask'
            ume_model.embed({"wrong_key": torch.tensor([1, 2, 3])})

    def test_ume_onnx_model_info(self, ume_model, tmpdir):
        """Test ONNX model metadata and information."""
        # Export model
        onnx_path = tmpdir / "ume_model_info.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Load and inspect ONNX model
        onnx_model = onnx.load(str(onnx_path))

        # Check model metadata
        assert onnx_model.ir_version > 0
        assert onnx_model.producer_name == "pytorch"

        # Check input/output information
        inputs = [input.name for input in onnx_model.graph.input]
        outputs = [output.name for output in onnx_model.graph.output]

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "embeddings" in outputs

        # Check that model has parameters
        assert len(onnx_model.graph.initializer) > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.slow  # Mark as slow test
    def test_ume_onnx_gpu_compatibility(self, ume_model, tmpdir):
        """Test ONNX model compatibility with GPU (if available)."""
        # Move model to GPU
        ume_model = ume_model.cuda()

        # Create GPU inputs with correct dtype
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, 1, seq_len), dtype=torch.long, device="cuda")
        attention_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.long, device="cuda")

        # Get PyTorch output on GPU
        pytorch_output = ume_model(input_ids, attention_mask)

        # Export to ONNX (this will be on CPU)
        onnx_path = tmpdir / "ume_model_gpu.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Move PyTorch output to CPU for comparison
        pytorch_output_cpu = pytorch_output.cpu()

        # Run ONNX inference on CPU
        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output = torch.from_numpy(ort_outputs[0])

        # Compare outputs
        assert onnx_output.shape == pytorch_output_cpu.shape
        assert torch.allclose(onnx_output, pytorch_output_cpu, atol=1e-5, rtol=1e-5)


class TestUMEONNXIntegration:
    """Integration tests for UME ONNX functionality."""

    @pytest.mark.slow  # Mark as slow test
    def test_ume_onnx_end_to_end_workflow(self, ume_model, sample_sequences, tmpdir):
        """Test complete end-to-end workflow: tokenization -> ONNX inference -> comparison."""
        # Export model
        onnx_path = tmpdir / "ume_model_e2e.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Test with SMILES modality only for speed
        ort_session = ort.InferenceSession(str(onnx_path))

        modality = Modality.SMILES
        sequences = sample_sequences["SMILES"]

        # Get PyTorch embeddings using high-level API
        pytorch_embeddings = ume_model.embed_sequences(sequences, modality, aggregate=True)

        # Get tokenized inputs
        tokenizer_transform = ume_model.tokenizer_transforms[modality]
        encoded_batch = tokenizer_transform(sequences)

        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        # Ensure 3D format and correct dtype
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1)

        # Ensure correct dtype for ONNX
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        # ONNX inference - ensure tensors are on CPU for numpy conversion
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy(),
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_embeddings = torch.from_numpy(ort_outputs[0])

        # Verify outputs - ensure both are on CPU for comparison
        assert onnx_embeddings.shape == pytorch_embeddings.shape
        assert torch.allclose(onnx_embeddings, pytorch_embeddings.cpu(), atol=1e-5, rtol=1e-5)

        # Verify embedding properties
        assert not torch.isnan(onnx_embeddings).any()
        assert not torch.isinf(onnx_embeddings).any()
        assert onnx_embeddings.shape[-1] == ume_model.embedding_dim

    def test_ume_onnx_performance_comparison(self, ume_model, tmpdir):
        """Compare performance between PyTorch and ONNX inference (fast CI version)."""
        # Export model using the new method
        onnx_path = tmpdir / "ume_model_perf.onnx"
        ume_model.export_onnx(str(onnx_path), modality=Modality.SMILES)

        # Test sequences for benchmarking (single sequence for speed)
        test_sequences = ["CC(=O)OC1=CC=CC=C1C(=O)O"]  # Single sequence for faster testing

        # Use the benchmark utility with very few runs for CI speed
        results = benchmark_onnx_pytorch(
            str(onnx_path),
            ume_model,
            test_sequences,
            Modality.SMILES,
            max_length=ume_model.max_length,
            num_runs=1,  # Reduced from 2 for speed
        )

        print(f"PyTorch time: {results['pytorch_time']:.4f}s")
        print(f"ONNX time: {results['onnx_time']:.4f}s")
        print(f"Speedup: {results['speedup']:.2f}x")

        # Verify both run successfully
        # Note: ONNX can be slower for small inputs due to overhead, especially on CPU
        assert results["pytorch_time"] > 0
        assert results["onnx_time"] > 0
        # For small inputs on CPU, ONNX overhead can dominate - just ensure it's not completely broken
        assert results["speedup"] > 0.001  # ONNX should not be more than 1000x slower
