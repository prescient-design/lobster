"""DGEB adapter for ESM models."""

import logging
from typing import Literal
from collections.abc import Callable

import numpy as np
import torch
import lightning as L
from dgeb.models import BioSeqTransformer
from dgeb.modality import Modality as DGEBModality

from lobster.constants import Modality
from ._pooling_utils import apply_dgeb_pooling, create_attention_mask_from_embeddings

logger = logging.getLogger(__name__)


class ESMAdapterDGEB(BioSeqTransformer):
    """Adapter to make ESM models compatible with DGEB evaluation framework.

    This adapter wraps ESM Lightning modules to provide the interface expected by DGEB,
    allowing evaluation on protein and DNA benchmarks without requiring checkpoints.

    Parameters
    ----------
    module : L.LightningModule
        The ESM Lightning module to wrap.
    modality : Modality
        Biological modality for the sequences. One of Modality.AMINO_ACID or Modality.NUCLEOTIDE.
    batch_size : int, default=32
        Batch size for encoding.
    max_seq_length : int, default=1024
        Maximum sequence length for input sequences.
    l2_norm : bool, default=False
        Whether to L2-normalize embeddings before returning.
    pool_type : str, default="mean"
        Pooling strategy. One of "mean", "max", "cls", "last".
    devices : list[int] | None, default=None
        List of device IDs to use for inference. If None, uses [0].
    layers : list[int] | Literal["mid"] | Literal["last"] | None, default="last"
        Layers to extract embeddings from. For ESM, typically only last layer is meaningful.
    process_and_embed_fn : Callable
        Function to process inputs and extract embeddings from the ESM model.
        Should have signature: (module, sequences, modality, aggregate) -> torch.Tensor
    """

    def __init__(
        self,
        module: L.LightningModule,
        modality: Modality = Modality.AMINO_ACID,
        batch_size: int = 32,
        max_seq_length: int = 1024,
        l2_norm: bool = False,
        pool_type: str = "mean",
        devices: list[int] | None = None,
        layers: list[int] | Literal["mid"] | Literal["last"] | None = "last",
        process_and_embed_fn: Callable | None = None,
    ):
        logger.info(f"Initializing ESMAdapterDGEB with modality={modality}")

        if process_and_embed_fn is None:
            raise ValueError("process_and_embed_fn must be provided for ESM adapter")

        # Store the ESM module and processing function
        self.esm_module = module
        self.process_and_embed_fn = process_and_embed_fn
        self._modality = modality
        self._max_seq_length = max_seq_length

        # Determine embedding dimension from a sample sequence
        logger.info("Determining embedding dimension...")
        try:
            sample_embedding = self.process_and_embed_fn(module, ["M"], modality="amino_acid", aggregate=True)
            self._embed_dim = sample_embedding.shape[-1]
            logger.info(f"Detected embedding dimension: {self._embed_dim}")
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            # Fallback to common ESM-2 embedding dimension
            self._embed_dim = 1280
            logger.info(f"Using fallback embedding dimension: {self._embed_dim}")

        # Set layers for ESM evaluation
        if layers is None:
            layers = "last"  # Default to last layer
        self.layers = layers

        # Initialize parent with required parameters
        super().__init__(
            model_name="ESM",
            layers=self.layers,
            devices=devices or [0],
            max_seq_length=max_seq_length,
            l2_norm=l2_norm,
            batch_size=batch_size,
            pool_type=pool_type,
        )

        logger.info("ESMAdapterDGEB initialization completed")

    def _load_model(self, model_name: str):
        """Override to use the provided ESM module instead of loading from checkpoint."""
        logger.info("Using provided ESM module, skipping model loading")

        # DGEB expects the model to have a config attribute
        if not hasattr(self.esm_module, "config"):
            # Create a minimal config object for DGEB compatibility
            from types import SimpleNamespace

            self.esm_module.config = SimpleNamespace(hidden_size=self._embed_dim, num_hidden_layers=1)
            logger.info(f"Added config to ESM module: hidden_size={self._embed_dim}, num_layers=1")

        return self.esm_module

    def _get_tokenizer(self, model_name: str):
        """Provide a minimal tokenizer for BioSeqTransformer interface compliance.

        Note: ESM sequence processing is handled entirely through process_and_embed_fn,
        so this tokenizer is only used to satisfy the parent class interface requirements.
        """

        # Create a dummy tokenizer object with required attributes
        class DummyTokenizer:
            model_max_length = self._max_seq_length
            vocab_size = 33  # ESM amino acid vocab size

            @staticmethod
            def encode(text, **kwargs):
                return list(range(len(text)))

            @staticmethod
            def decode(token_ids, **kwargs):
                return ""

            @staticmethod
            def get_vocab():
                return {}

        return DummyTokenizer()

    def encode(self, sequences: list[str], **kwargs) -> np.ndarray:
        """Encode sequences to embeddings using the ESM model.

        Parameters
        ----------
        sequences : list[str]
            List of biological sequences to encode.
        **kwargs
            Additional keyword arguments (unused but required by DGEB interface).

        Returns
        -------
        np.ndarray
            Embeddings array of shape [num_sequences, num_layers, embedding_dim].
        """
        if not sequences:
            return np.array([])

        # Filter out empty sequences
        valid_sequences = [seq for seq in sequences if seq and seq.strip()]
        if not valid_sequences:
            logger.warning("No valid sequences to encode")
            return np.empty((0, 1, self._embed_dim))

        # Process sequences in batches
        all_embeddings = []

        for i in range(0, len(valid_sequences), self.batch_size):
            batch_sequences = valid_sequences[i : i + self.batch_size]

            try:
                # Get token-level embeddings first (no aggregation)
                token_embeddings = self.process_and_embed_fn(
                    self.esm_module, batch_sequences, modality="amino_acid", aggregate=False
                )

                # Apply proper masked pooling (similar to UME adapter)
                pooled_embeddings = self._apply_masked_pooling(token_embeddings, batch_sequences)

                # Convert to numpy and ensure proper shape
                batch_embeddings_np = pooled_embeddings.detach().cpu().numpy()
                if batch_embeddings_np.ndim == 2:
                    # Add layer dimension: (batch_size, embed_dim) -> (batch_size, 1, embed_dim)
                    batch_embeddings_np = batch_embeddings_np[:, np.newaxis, :]

                all_embeddings.append(batch_embeddings_np)

            except Exception as e:
                batch_num = i // self.batch_size + 1
                logger.error(f"Error encoding batch {batch_num}: {e}")
                # Create zero embeddings for failed batch to maintain shape consistency
                batch_size_local = len(batch_sequences)
                zero_embeddings = np.zeros((batch_size_local, 1, self._embed_dim))
                all_embeddings.append(zero_embeddings)
                logger.warning(
                    f"Returning zero embeddings for {batch_size_local} sequences in failed batch {batch_num}"
                )

        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)

        # Apply L2 normalization if requested (same as UME DGEB adapter)
        if self.l2_norm:
            # Normalize across the embedding dimension for each layer (axis=2)
            norms = np.linalg.norm(embeddings, axis=2, keepdims=True)
            embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)

        logger.info(f"Encoded {len(valid_sequences)} sequences to embeddings of shape {embeddings.shape}")
        return embeddings

    def _apply_masked_pooling(self, token_embeddings: torch.Tensor, sequences: list[str]) -> torch.Tensor:
        """Apply proper masked pooling to token-level embeddings.

        Uses shared pooling utilities to ensure consistency with UME adapter.

        Parameters
        ----------
        token_embeddings : torch.Tensor
            Token-level embeddings, shape depends on ESM output format
        sequences : list[str]
            Original sequences for creating attention masks

        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape (batch_size, embedding_dim)
        """
        if isinstance(token_embeddings, list):
            # Handle variable-length sequences (when batch can't be stacked)
            pooled_embeddings = []
            for seq_emb in token_embeddings:
                if seq_emb.dim() == 2:  # (seq_len, hidden_size)
                    seq_emb = seq_emb.unsqueeze(0)  # (1, seq_len, hidden_size)

                # Create attention mask from embedding magnitudes
                attention_mask = create_attention_mask_from_embeddings(seq_emb)

                # Apply pooling using shared utility
                pooled = apply_dgeb_pooling(seq_emb, attention_mask, self.pool_type)
                pooled_embeddings.append(pooled)

            return torch.cat(pooled_embeddings, dim=0)

        else:
            # Handle batched embeddings (batch_size, seq_len, hidden_size)
            if token_embeddings.dim() == 2:
                # Single sequence: (seq_len, hidden_size) -> (1, seq_len, hidden_size)
                token_embeddings = token_embeddings.unsqueeze(0)

            # Create attention mask from embedding magnitudes
            attention_mask = create_attention_mask_from_embeddings(token_embeddings)

            # Apply pooling using shared utility
            return apply_dgeb_pooling(token_embeddings, attention_mask, self.pool_type)

    @property
    def modality(self) -> DGEBModality:
        """Return the biological modality (DGEB enum)."""
        modality_map = {
            Modality.AMINO_ACID: DGEBModality.PROTEIN,
            Modality.NUCLEOTIDE: DGEBModality.DNA,
        }

        if self._modality not in modality_map:
            raise ValueError(f"Unsupported modality: {self._modality}. Supported: {list(modality_map.keys())}")

        return modality_map[self._modality]

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embed_dim

    @property
    def num_layers(self) -> int:
        """Return the number of layers being extracted."""
        if isinstance(self.layers, list):
            return len(self.layers)
        else:
            return 1  # For "last", "mid", etc.

    @property
    def metadata(self) -> dict:
        """Return model metadata (compatible with DGEB expectations)."""
        # Determine current device and parameter count
        try:
            device = next(self.esm_module.parameters()).device.type
            total_params = sum(p.numel() for p in self.esm_module.parameters())
        except (StopIteration, AttributeError):
            # Fallback if model has no parameters or is not properly initialized
            device = "cpu"
            total_params = 0

        return {
            "model_name": "ESM",
            "hf_name": "ESM",  # Required by DGEB
            "modality": self._modality.value,  # Use string value for compatibility
            "embed_dim": self._embed_dim,  # Required by DGEB
            "num_layers": self.num_layers,  # Required by DGEB
            "layers": self.layers,
            "num_params": total_params,
            "max_seq_length": self._max_seq_length,
            "pool_type": self.pool_type,
            "l2_norm": self.l2_norm,
            "batch_size": self.batch_size,
            "device": device,
            "use_flash_attn": False,  # ESM doesn't use flash attn through this interface
            "cuda_available": torch.cuda.is_available(),
        }
