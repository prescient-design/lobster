"""ONNX utilities for UME model inference."""

import logging

import numpy as np
import onnxruntime as ort

from lobster.constants import Modality, ModalityType
from lobster.tokenization import UMETokenizerTransform

logger = logging.getLogger(__name__)


def run_onnx_inference(
    onnx_path: str,
    sequences: str | list[str],
    modality: ModalityType | Modality | None = None,
    max_length: int = 8192,
    session_options: ort.SessionOptions | None = None,
) -> np.ndarray:
    """Run ONNX inference for UME model with actual sequences.

    This function handles tokenization and inference in one call, providing
    a complete end-to-end experience for ONNX inference.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file.
    sequences : Union[str, List[str]]
        Input sequences to embed. Can be a single string or list of strings.
    modality : ModalityType | Modality | None, optional
        Modality of the input sequences. If None, the tokenizer will auto-detect
        the modality for each sequence.
    max_length : int, default=8192
        Maximum sequence length for tokenization.
    session_options : ort.SessionOptions | None, optional
        ONNX Runtime session options. If None, uses default options.

    Returns
    -------
    np.ndarray
        Output embeddings of shape (batch_size, embedding_dim).

    Examples
    --------
    >>> from lobster.model.onnx_utils import run_onnx_inference
    >>>
    >>> # Single SMILES sequence (auto-detected)
    >>> embeddings = run_onnx_inference(
    ...     "ume_universal.onnx",
    ...     "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    ... )
    >>> print(embeddings.shape)  # (1, 768)
    >>>
    >>> # Multiple mixed sequences (auto-detected)
    >>> mixed_sequences = [
    ...     "MKTVRQERLKSIVRILERSKEPVSGAQL",  # Protein
    ...     "CC(=O)O",  # SMILES
    ...     "ATGCATGC"  # DNA
    ... ]
    >>> embeddings = run_onnx_inference(
    ...     "ume_universal.onnx",
    ...     mixed_sequences
    ... )
    >>> print(embeddings.shape)  # (3, 768)
    """
    # Convert single string to list
    if isinstance(sequences, str):
        sequences = [sequences]

    # Set up session options
    if session_options is None:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1

    # Create ONNX session
    ort_session = ort.InferenceSession(onnx_path, session_options)

    # Tokenize sequences - auto-detect modality if not specified
    tokenizer_transform = UMETokenizerTransform(
        max_length=max_length, return_modality=True, padding="max_length", add_special_tokens=True
    )
    encoded_batch = tokenizer_transform(sequences, modality=modality)

    input_ids = encoded_batch["input_ids"]
    attention_mask = encoded_batch["attention_mask"]

    # Ensure 3D format for ONNX (batch_size, 1, sequence_length)
    if input_ids.dim() == 2:
        input_ids = input_ids.unsqueeze(1)
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.unsqueeze(1)

    # Ensure correct dtype for ONNX
    input_ids = input_ids.long()
    attention_mask = attention_mask.long()

    # Prepare inputs for ONNX Runtime
    ort_inputs = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    }

    # Run inference
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs[0]


def compare_onnx_pytorch(
    onnx_path: str,
    pytorch_model,
    sequences: str | list[str],
    modality: ModalityType | Modality | None = None,
    max_length: int = 8192,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """Compare ONNX and PyTorch outputs for the same input sequences.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file.
    pytorch_model
        PyTorch UME model instance.
    sequences : Union[str, List[str]]
        Input sequences to compare.
    modality : ModalityType | Modality | None, optional
        Modality of the input sequences. If None, the tokenizer will auto-detect
        the modality for each sequence.
    max_length : int, default=8192
        Maximum sequence length for tokenization.
    atol : float, default=1e-5
        Absolute tolerance for comparison.
    rtol : float, default=1e-5
        Relative tolerance for comparison.

    Returns
    -------
    bool
        True if outputs match within tolerance, False otherwise.

    Examples
    --------
    >>> from lobster.model import UME
    >>> from lobster.model.onnx_utils import compare_onnx_pytorch
    >>>
    >>> # Initialize PyTorch model
    >>> ume = UME(model_name="UME_mini")
    >>>
    >>> # Compare outputs with auto-detected modalities
    >>> sequences = ["CC(=O)OC1=CC=CC=C1C(=O)O", "MKTVRQERLKSIVRILERSKEPVSGAQL"]
    >>> match = compare_onnx_pytorch("ume_universal.onnx", ume, sequences)
    >>> print(f"Outputs match: {match}")
    """
    # Use the model's max_length if available and not specified
    if hasattr(pytorch_model, "max_length") and max_length == 8192:
        max_length = pytorch_model.max_length

    # Get ONNX output
    onnx_output = run_onnx_inference(onnx_path, sequences, modality, max_length)

    # Get PyTorch output
    pytorch_output = pytorch_model.embed_sequences(sequences, modality, aggregate=True)
    pytorch_output_np = pytorch_output.cpu().numpy()

    # Compare outputs
    try:
        np.testing.assert_allclose(onnx_output, pytorch_output_np, atol=atol, rtol=rtol)
        return True
    except AssertionError as e:
        logger.warning(f"ONNX and PyTorch outputs don't match: {e}")
        return False


def benchmark_onnx_pytorch(
    onnx_path: str,
    pytorch_model,
    sequences: str | list[str],
    modality: ModalityType | Modality,
    max_length: int = 8192,
    num_runs: int = 10,
) -> dict:
    """Benchmark ONNX vs PyTorch inference performance.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model file.
    pytorch_model
        PyTorch UME model instance.
    sequences : Union[str, List[str]]
        Input sequences to benchmark.
    modality : ModalityType | Modality
        Modality of the input sequences.
    max_length : int, default=8192
        Maximum sequence length for tokenization.
    num_runs : int, default=10
        Number of runs for timing.

    Returns
    -------
    dict
        Dictionary with timing results and speedup information.

    Examples
    --------
    >>> from lobster.model import UME
    >>> from lobster.model.onnx_utils import benchmark_onnx_pytorch
    >>> from lobster.constants import Modality
    >>>
    >>> # Initialize PyTorch model
    >>> ume = UME(model_name="UME_mini")
    >>>
    >>> # Benchmark performance
    >>> sequences = ["CC(=O)OC1=CC=CC=C1C(=O)O"] * 10  # 10 copies for batch
    >>> results = benchmark_onnx_pytorch("ume_smiles.onnx", ume, sequences, Modality.SMILES)
    >>> print(f"PyTorch time: {results['pytorch_time']:.4f}s")
    >>> print(f"ONNX time: {results['onnx_time']:.4f}s")
    >>> print(f"Speedup: {results['speedup']:.2f}x")
    """
    import time

    # Convert single string to list
    if isinstance(sequences, str):
        sequences = [sequences]

    # Use the model's max_length if available and not specified
    if hasattr(pytorch_model, "max_length") and max_length == 8192:
        max_length = pytorch_model.max_length

    # Warm up - run once for basic initialization
    _ = run_onnx_inference(onnx_path, sequences, modality, max_length)
    _ = pytorch_model.embed_sequences(sequences, modality, aggregate=True)

    # PyTorch timing
    start_time = time.time()
    for _ in range(num_runs):
        _ = pytorch_model.embed_sequences(sequences, modality, aggregate=True)
    pytorch_time = time.time() - start_time

    # ONNX timing
    start_time = time.time()
    for _ in range(num_runs):
        _ = run_onnx_inference(onnx_path, sequences, modality, max_length)
    onnx_time = time.time() - start_time

    return {
        "pytorch_time": pytorch_time / num_runs,
        "onnx_time": onnx_time / num_runs,
        "speedup": pytorch_time / onnx_time,
        "num_runs": num_runs,
        "batch_size": len(sequences),
    }
