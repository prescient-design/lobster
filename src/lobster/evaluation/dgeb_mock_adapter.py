"""Mock DGEB adapter that returns random embeddings for baseline evaluation."""

import logging
from typing import Literal

import numpy as np
import torch
from dgeb.models import BioSeqTransformer
from dgeb.modality import Modality

logger = logging.getLogger(__name__)


class MockUMEAdapterDGEB(BioSeqTransformer):
    """Mock adapter class that returns random embeddings for DGEB evaluation.

    This adapter provides the same interface as UMEAdapterDGEB but returns
    completely random embeddings, serving as a random baseline for all tasks.

    Parameters
    ----------
    model_name : str
        Name of the mock model (used for metadata only).
    layers : list[int] | Literal["mid"] | Literal["last"] | None, default=None
        Which layers to simulate. If None, uses last layer.
    devices : list[int] | None, default=None
        List of device IDs (unused in mock implementation).
    num_processes : int, default=16
        Number of processes (unused in mock implementation).
    max_seq_length : int, default=1024
        Maximum sequence length (used for metadata only).
    l2_norm : bool, default=False
        Whether to L2-normalize embeddings before returning.
    batch_size : int, default=128
        Batch size (unused in mock implementation).
    pool_type : str, default="mean"
        Pooling strategy (used for metadata only).
    modality : Literal["protein", "dna"], default="protein"
        Biological modality for the sequences.
    use_flash_attn : bool | None, default=None
        Whether to use flash attention (unused in mock implementation).
    embed_dim : int, default=768
        Embedding dimension for the random embeddings.
    num_layers : int, default=12
        Number of layers to simulate.
    seed : int, default=42
        Random seed for reproducible random embeddings.
    """

    def __init__(
        self,
        model_name: str,
        layers: list[int] | Literal["mid"] | Literal["last"] | None = None,
        devices: list[int] | None = None,
        num_processes: int = 16,
        max_seq_length: int = 1024,
        l2_norm: bool = False,
        batch_size: int = 128,
        pool_type: str = "mean",
        modality: Literal["protein", "dna"] = "protein",
        use_flash_attn: bool | None = None,
        embed_dim: int = 768,
        num_layers: int = 12,
        seed: int = 42,
    ):
        logger.info(f"Initializing MockUMEAdapterDGEB with model_name={model_name}, modality={modality}")

        # Handle devices parameter
        if devices is None:
            devices = [0]

        # Store configuration before calling parent constructor
        self._modality = modality
        self._model_name = model_name
        self._embed_dim = embed_dim
        self._num_layers = num_layers
        self._use_flash_attn = use_flash_attn
        self._devices = devices
        self._seed = seed

        # Set hf_name as regular attribute (not property) for DGEB compatibility
        self.hf_name = model_name

        # Call parent constructor to set up required attributes
        logger.info("Calling parent BioSeqTransformer.__init__...")
        super().__init__(
            model_name=model_name,
            layers=layers,
            devices=devices,
            num_processes=num_processes,
            max_seq_length=max_seq_length,
            l2_norm=l2_norm,
            batch_size=batch_size,
            pool_type=pool_type,
        )
        logger.info("Parent constructor completed")

        # Set random seed for reproducible results
        np.random.seed(seed)
        torch.manual_seed(seed)

        logger.info(f"Mock adapter configured with {len(self.layers)} layers: {self.layers}")
        logger.info(f"Embedding dimension: {embed_dim}, Max sequence length: {max_seq_length}")

    def _load_model(self, model_name: str):
        """Override model loading to prevent HuggingFace Hub calls for mock adapter.

        This method is called by the parent BioSeqTransformer constructor.
        For the mock adapter, we create a mock model object with the required
        attributes to satisfy the parent class requirements.

        Parameters
        ----------
        model_name : str
            Model name (unused in mock implementation).

        Returns
        -------
        MockModel
            A mock model object with required attributes.
        """
        logger.info(f"Mock adapter: Creating mock model for '{model_name}'")
        logger.info("Mock adapter generates random embeddings - no real model needed")

        # Create a mock model object with the required attributes
        class MockModel:
            def __init__(self, embed_dim, num_layers):
                self.embedding_dim = embed_dim
                self.config = MockConfig(num_layers)

            def parameters(self):
                # Return empty parameter iterator
                return iter([])

            def freeze(self):
                # Mock freeze method
                pass

            def cuda(self):
                # Mock cuda method - return self
                return self

            def to(self, device):
                # Mock to method - return self (device doesn't matter for mock)
                return self

            def eval(self):
                # Mock eval method - return self (evaluation mode doesn't matter for mock)
                return self

        class MockConfig:
            def __init__(self, num_layers):
                self.num_hidden_layers = num_layers

        # Create and return the mock model
        mock_model = MockModel(self._embed_dim, self._num_layers)
        logger.info(f"Created mock model with embedding_dim={self._embed_dim}, num_layers={self._num_layers}")

        return mock_model

    def _get_tokenizer(self, model_name: str):
        """Override tokenizer loading to prevent HuggingFace Hub calls for mock adapter.

        This method is called by the parent BioSeqTransformer constructor.
        For the mock adapter, we create a mock tokenizer since we don't need
        real tokenization for random embeddings.

        Parameters
        ----------
        model_name : str
            Model name (unused in mock implementation).

        Returns
        -------
        MockTokenizer
            A mock tokenizer object with required methods.
        """
        logger.info(f"Mock adapter: Creating mock tokenizer for '{model_name}'")
        logger.info("Mock adapter generates random embeddings - no real tokenization needed")

        # Create a mock tokenizer object with the required methods
        class MockTokenizer:
            def __init__(self):
                self.model_max_length = 1024
                self.vocab_size = 50000  # Reasonable default

            def encode(self, text, **kwargs):
                # Simple character-based encoding as fallback
                _ = kwargs  # Suppress unused parameter warning
                return list(range(len(text)))

            def decode(self, token_ids, **kwargs):
                # Simple decoding fallback
                _ = kwargs  # Suppress unused parameter warning
                return "".join([chr(65 + (i % 26)) for i in token_ids])

            def get_vocab(self):
                return {}

        # Create and return the mock tokenizer
        mock_tokenizer = MockTokenizer()
        logger.info(f"Created mock tokenizer with vocab_size={mock_tokenizer.vocab_size}")

        return mock_tokenizer

    def encode(self, sequences: list[str], **kwargs) -> np.ndarray:
        """Encode sequences to random embeddings.

        Parameters
        ----------
        sequences : list[str]
            List of biological sequences to encode.
        **kwargs
            Additional keyword arguments (unused but required by DGEB interface).

        Returns
        -------
        np.ndarray
            Random embeddings array of shape [num_sequences, num_layers, embedding_dim].
        """
        _ = kwargs  # Suppress unused parameter warning

        if not sequences:
            return np.array([])

        # Filter out empty sequences
        valid_sequences = [seq for seq in sequences if seq.strip()]
        if not valid_sequences:
            logger.warning("No valid sequences to encode")
            return np.array([])

        # Generate random embeddings
        num_sequences = len(valid_sequences)
        embeddings = np.random.randn(num_sequences, len(self.layers), self._embed_dim).astype(np.float32)

        # Apply L2 normalization if requested
        if self.l2_norm:
            # Normalize across the embedding dimension for each layer
            embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        logger.info(f"Generated random embeddings for {num_sequences} sequences with shape {embeddings.shape}")
        return embeddings

    @property
    def modality(self) -> Modality:
        """Return the biological modality."""
        if self._modality == "protein":
            return Modality.PROTEIN
        elif self._modality == "dna":
            return Modality.DNA
        else:
            raise ValueError(f"Unsupported modality: {self._modality}")

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embed_dim

    @property
    def num_layers(self) -> int:
        """Return the number of layers."""
        return self._num_layers

    @property
    def metadata(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": self._model_name,
            "hf_name": self._model_name,  # Required by DGEB
            "modality": self._modality,
            "embed_dim": self._embed_dim,  # Required by DGEB
            "num_layers": self._num_layers,  # Required by DGEB
            "num_params": 0,  # Mock model has no parameters
            "max_seq_length": self.max_seq_length,
            "pool_type": self.pool_type,
            "l2_norm": self.l2_norm,
            "batch_size": self.batch_size,
            "device": "cpu",  # Mock model doesn't use GPU
            "use_flash_attn": self._use_flash_attn,
            "cuda_available": torch.cuda.is_available(),
            "is_mock": True,  # Flag to indicate this is a mock model
            "random_seed": self._seed,  # Include seed for reproducibility
        }
