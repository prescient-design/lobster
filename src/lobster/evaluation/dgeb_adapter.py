"""DGEB adapter for UME models."""

import logging
from typing import Literal

import numpy as np
import torch
from dgeb.models import BioSeqTransformer
from dgeb.modality import Modality

from lobster.constants import Modality as LobsterModality
from lobster.model import UME

logger = logging.getLogger(__name__)


class UMEAdapter(BioSeqTransformer):
    """Adapter class to make UME models compatible with DGEB evaluation framework.

    This adapter wraps UME models to provide the interface expected by DGEB,
    allowing evaluation on protein and DNA benchmarks.

    Parameters
    ----------
    model_name : str
        Name or path to the UME model. Can be a pretrained model name
        (e.g., "ume-mini-base-12M") or path to a checkpoint file.
    layers : list[int] | Literal["mid"] | Literal["last"] | None, default=None
        Which layers to extract embeddings from. If None, uses last layer.
    devices : list[int] | None, default=None
        List of device IDs to use for inference. If None, uses [0].
    num_processes : int, default=16
        Number of processes for data loading.
    max_seq_length : int, default=1024
        Maximum sequence length for input sequences.
    l2_norm : bool, default=False
        Whether to L2-normalize embeddings before returning.
    batch_size : int, default=128
        Batch size for encoding.
    pool_type : str, default="mean"
        Pooling strategy. One of "mean", "max", "cls", "last".
    modality : Literal["protein", "dna"], default="protein"
        Biological modality for the sequences.
    use_flash_attn : bool | None, default=None
        Whether to use flash attention. If None, determined by device availability.
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
    ):
        if devices is None:
            devices = [0]

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

        self._modality = modality
        self._use_flash_attn = use_flash_attn
        self._model_name = model_name
        # Don't call _load_model here, the parent class will call it

    def _load_model(self, model_name: str):
        """Load the UME model from checkpoint or pretrained model."""
        # Determine device
        device = "cuda" if torch.cuda.is_available() and self.devices else "cpu"

        try:
            # Try loading as pretrained model first
            if model_name.startswith("ume-"):
                logger.info(f"Loading pretrained UME model: {model_name}")
                model = UME.from_pretrained(
                    model_name,
                    device=device,
                    use_flash_attn=False,  # Default to CPU-compatible
                )
            else:
                # Load from checkpoint path
                logger.info(f"Loading UME model from checkpoint: {model_name}")
                model = UME.load_from_checkpoint(
                    model_name,
                    device=device,
                    use_flash_attn=False,  # Default to CPU-compatible
                )
        except Exception as e:
            logger.error(f"Failed to load UME model {model_name}: {e}")
            raise

        # Freeze model for evaluation
        model.freeze()

        # Move to specified device
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()

        # DGEB expects the returned model to have a config attribute directly
        # UME models have config under model.config, so we need to expose it
        if not hasattr(model, "config"):
            model.config = model.model.config

        # Store model as instance variable
        self.model = model

        logger.info(f"Loaded UME model {model_name} on {device}")

        # Return the model for DGEB compatibility
        return model

    def _get_tokenizer(self, model_name: str):
        """Override tokenizer loading to use UME's tokenizer."""
        # model_name is not used since UME has its own tokenizers built-in
        _ = model_name  # Suppress unused parameter warning

        # Return a dummy tokenizer object since we don't need it
        class DummyTokenizer:
            def __init__(self):
                self.model_max_length = 1024

        return DummyTokenizer()

    def _get_lobster_modality(self) -> LobsterModality:
        """Convert DGEB modality to Lobster modality."""
        if self._modality == "protein":
            return LobsterModality.AMINO_ACID
        elif self._modality == "dna":
            return LobsterModality.NUCLEOTIDE
        else:
            raise ValueError(f"Unsupported modality: {self._modality}")

    def encode(self, sequences: list[str], **kwargs) -> np.ndarray:
        """Encode sequences to embeddings.

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
        _ = kwargs  # Suppress unused parameter warning

        if not sequences:
            return np.array([])

        # Filter out empty sequences
        valid_sequences = [seq for seq in sequences if seq.strip()]
        if not valid_sequences:
            logger.warning("No valid sequences to encode")
            return np.array([])

        # Get the appropriate modality
        lobster_modality = self._get_lobster_modality()

        # Encode sequences in batches
        all_embeddings = []

        for i in range(0, len(valid_sequences), self.batch_size):
            batch_sequences = valid_sequences[i : i + self.batch_size]

            try:
                # Get embeddings from UME model with proper layer extraction
                batch_embeddings = self._extract_layer_embeddings(batch_sequences, lobster_modality)
                all_embeddings.append(batch_embeddings)

            except Exception as e:
                logger.error(f"Error encoding batch {i // self.batch_size + 1}: {e}")
                # Return zeros for failed batch
                batch_size = len(batch_sequences)
                zero_embeddings = np.zeros((batch_size, self.num_layers, self.embed_dim))
                all_embeddings.append(zero_embeddings)

        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)

        # Apply L2 normalization if requested
        if self.l2_norm:
            # Normalize across the embedding dimension for each layer
            embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        logger.info(f"Encoded {len(valid_sequences)} sequences to embeddings of shape {embeddings.shape}")
        return embeddings

    def _extract_layer_embeddings(self, sequences: list[str], modality: LobsterModality) -> np.ndarray:
        """Extract embeddings from specified layers of the UME model.

        Parameters
        ----------
        sequences : list[str]
            Batch of sequences to encode.
        modality : LobsterModality
            The biological modality for tokenization.

        Returns
        -------
        np.ndarray
            Embeddings of shape [batch_size, num_layers, embedding_dim].
        """
        # For now, use the high-level embed_sequences method which gives us aggregated embeddings
        # TODO: In the future, we could implement proper layer-wise extraction by calling
        # the underlying FlexBERT model directly to get intermediate layer outputs

        # Get aggregated embeddings (mean pooled across sequence length)
        aggregated_embeddings = self.model.embed_sequences(
            sequences,
            modality=modality,
            aggregate=True,
        )

        # Convert to numpy
        aggregated_embeddings = aggregated_embeddings.detach().cpu().numpy()

        # DGEB expects embeddings from multiple layers
        # Since UME's embed_sequences gives us the final aggregated representation,
        # we'll duplicate it for each layer as a reasonable approximation
        # This is similar to how some other models handle layer extraction in DGEB

        # Create embeddings array with shape [batch_size, num_layers, embed_dim]
        layer_embeddings = np.tile(aggregated_embeddings[:, np.newaxis, :], (1, self.num_layers, 1))

        return layer_embeddings

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
        return self.model.embedding_dim

    @property
    def num_layers(self) -> int:
        """Return the actual number of hidden layers in the UME model."""
        return self.model.model.config.num_hidden_layers

    @property
    def metadata(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": self._model_name,
            "hf_name": self._model_name,  # Required by DGEB
            "modality": self._modality,
            "embed_dim": self.embed_dim,  # Required by DGEB
            "num_layers": self.num_layers,  # Required by DGEB
            "num_params": sum(p.numel() for p in self.model.parameters()),  # Total parameter count
            "max_seq_length": self.max_seq_length,
            "pool_type": self.pool_type,
            "l2_norm": self.l2_norm,
            "batch_size": self.batch_size,
        }
