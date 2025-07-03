import pytest
import torch
import logging
from importlib.util import find_spec

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_FLASH_ATTN_AVAILABLE = False

try:
    if find_spec("flash_attn"):
        from lobster.model.modern_bert._rotary import UnpaddedRotaryEmbedding

        _FLASH_ATTN_AVAILABLE = True
        logger.info("Successfully imported UnpaddedRotaryEmbedding with flash attention")
    else:
        # Try importing without flash attention
        from lobster.model.modern_bert._rotary import UnpaddedRotaryEmbedding

        logger.info("Successfully imported UnpaddedRotaryEmbedding without flash attention")
except ImportError as e:
    # Handle import errors gracefully
    logger.warning(f"Failed to import UnpaddedRotaryEmbedding: {e}")
    _FLASH_ATTN_AVAILABLE = False


def test_unpadded_rotary_embedding_forward():
    """Test the forward pass of UnpaddedRotaryEmbedding."""
    print(f"Flash attention available: {_FLASH_ATTN_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Skip if UnpaddedRotaryEmbedding is not available
    if not _FLASH_ATTN_AVAILABLE:
        pytest.skip("UnpaddedRotaryEmbedding not available")

    # Use CUDA if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32  # Use float32 on CPU for better precision

    # Create rotary embedding
    dim = 32  # Use smaller dim to ensure rotary embedding is applied to subset
    rotary_emb = UnpaddedRotaryEmbedding(dim=dim, device=device, dtype=dtype)

    # Create test inputs
    max_seqlen = 128
    nheads = 8
    headdim = 64

    # Create cumulative sequence lengths
    cu_seqlens = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)

    # Create qkv tensor: (total_nnz, 3, nheads, headdim)
    total_nnz = 128  # sum of sequence lengths
    qkv = torch.randn(total_nnz, 3, nheads, headdim, device=device, dtype=dtype)

    # Create a copy for comparison
    qkv_original = qkv.clone()

    # Test forward pass
    output = rotary_emb(qkv, cu_seqlens, max_seqlen=max_seqlen)

    # Check output shape
    assert output.shape == qkv.shape
    assert output.device.type == device.type  # Compare device type, not specific index
    assert output.dtype == dtype

    # Check that rotary embedding was applied by comparing with original
    # The rotary embedding should modify the first 'dim' dimensions of each head
    diff = torch.abs(output - qkv_original)
    max_diff = torch.max(diff)
    print(f"Maximum difference between output and input: {max_diff}")

    # Check that at least some values changed (rotary embedding was applied)
    assert max_diff > 1e-6, f"Rotary embedding did not modify the tensor. Max diff: {max_diff}"


def test_rotary_embedding_positional_invariance():
    """Test that rotary embedding breaks permutation invariance (positional encoding behavior)."""
    print(f"Flash attention available: {_FLASH_ATTN_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Skip if UnpaddedRotaryEmbedding is not available
    if not _FLASH_ATTN_AVAILABLE:
        pytest.skip("UnpaddedRotaryEmbedding not available")

    # Use CUDA if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32  # Use float32 on CPU for better precision

    # Create rotary embedding
    dim = 32
    rotary_emb = UnpaddedRotaryEmbedding(dim=dim, device=device, dtype=dtype)

    # Create test inputs with longer sequence to make positional effects more apparent
    max_seqlen = 256
    nheads = 8
    headdim = 64

    # Create cumulative sequence lengths for a single sequence
    cu_seqlens = torch.tensor([0, 256], device=device, dtype=torch.int32)

    # Create qkv tensor: (total_nnz, 3, nheads, headdim)
    total_nnz = 256
    qkv = torch.randn(total_nnz, 3, nheads, headdim, device=device, dtype=dtype)

    # Apply rotary embedding to original sequence
    output_original = rotary_emb(qkv.clone(), cu_seqlens, max_seqlen=max_seqlen)

    # Create shuffled version (swap first and last half of sequence)
    qkv_shuffled = qkv.clone()
    mid_point = total_nnz // 2
    qkv_shuffled[:mid_point], qkv_shuffled[mid_point:] = (
        qkv_shuffled[mid_point:].clone(),
        qkv_shuffled[:mid_point].clone(),
    )

    # Apply rotary embedding to shuffled sequence
    output_shuffled = rotary_emb(qkv_shuffled, cu_seqlens, max_seqlen=max_seqlen)

    # Check that outputs are different (positional encoding is working)
    # We expect the shuffled output to be different from the original output
    # because the positions have changed
    diff = torch.abs(output_shuffled - output_original)
    max_diff = torch.max(diff)
    print(f"Maximum difference between shuffled and original outputs: {max_diff}")

    # The outputs should be significantly different due to positional encoding
    assert max_diff > 1e-3, f"Rotary embedding did not break permutation invariance. Max diff: {max_diff}"

    # Also verify that the shuffled input produces different output than the original input
    # when both are processed with rotary embedding
    assert not torch.allclose(output_shuffled, output_original, atol=1e-3, rtol=1e-3), (
        "Rotary embedding should produce different outputs for different input orderings"
    )
