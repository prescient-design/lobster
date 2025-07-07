"""DGEB adapter for UME models."""

import logging
from typing import Literal

import numpy as np
import torch
from dgeb.models import BioSeqTransformer
from dgeb.modality import Modality

from lobster.constants import Modality as LobsterModality
from lobster.model import UME
from lobster.model.modern_bert._padding import unpad_input, pad_input

logger = logging.getLogger(__name__)


class UMEAdapterDGEB(BioSeqTransformer):
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
        logger.info(f"Initializing UMEAdapterDGEB with model_name={model_name}, modality={modality}")
        if devices is None:
            devices = [0]

        # Set attributes before calling parent __init__ since parent calls _load_model
        self._modality = modality
        self._use_flash_attn = use_flash_attn
        self._model_name = model_name
        self._devices = devices

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
        logger.info("UMEAdapterDGEB initialization completed")

    def _determine_optimal_settings(self) -> tuple[str, bool]:
        """Determine optimal device and flash attention settings.

        Returns
        -------
        tuple[str, bool]
            Device string and whether to use flash attention.
        """
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        has_gpu_devices = self._devices and any(d >= 0 for d in self._devices)

        if cuda_available and has_gpu_devices:
            device = "cuda"
            logger.info("CUDA detected - using GPU acceleration")
        else:
            device = "cpu"
            if not cuda_available:
                logger.info("CUDA not available - falling back to CPU")
            else:
                logger.info("No GPU devices specified - using CPU")

        # Determine flash attention usage
        use_flash_attn = False
        if self._use_flash_attn is not None:
            # User explicitly specified flash attention preference
            use_flash_attn = self._use_flash_attn
            if use_flash_attn and device == "cpu":
                logger.warning("Flash attention requested but using CPU - flash attention will be disabled")
                use_flash_attn = False
        elif device == "cuda":
            # Auto-detect flash attention availability on GPU
            use_flash_attn = self._check_flash_attention_available()
            if use_flash_attn:
                logger.info("Flash attention detected and enabled for better performance")
            else:
                logger.info("Flash attention not available - using standard attention")

        return device, use_flash_attn

    def _check_flash_attention_available(self) -> bool:
        """Check if flash attention is available.

        Returns
        -------
        bool
            True if flash attention is available and can be used.
        """
        import importlib.util

        # Check if flash_attn module is available without importing it
        flash_attn_spec = importlib.util.find_spec("flash_attn")
        if flash_attn_spec is None:
            logger.debug("Flash attention library not available")
            return False

        # Check if the specific flash_attn_func is available
        try:
            # Only import the specific function we need to test functionality
            from flash_attn import flash_attn_func

            # Test that we can access the function
            _ = flash_attn_func  # Suppress unused import warning
            logger.debug("Flash attention library found and functional")
            return True
        except (ImportError, AttributeError) as e:
            logger.debug(f"Flash attention components not available: {e}")
            return False
        except Exception as e:
            logger.debug(f"Flash attention check failed: {e}")
            return False

    def _load_model(self, model_name: str):
        """Load the UME model from checkpoint or pretrained model."""
        logger.info("Starting model loading process...")
        # Determine device and flash attention availability
        device, use_flash_attn = self._determine_optimal_settings()
        logger.info(f"Determined settings: device={device}, use_flash_attn={use_flash_attn}")

        try:
            # Try loading as pretrained model first
            if model_name.startswith("ume-"):
                logger.info(f"Loading pretrained UME model: {model_name}")
                logger.info("Calling UME.from_pretrained...")
                try:
                    model = UME.from_pretrained(
                        model_name,
                        device=device,
                        use_flash_attn=use_flash_attn,
                    )
                    logger.info("UME.from_pretrained completed successfully")
                except NotImplementedError as e:
                    logger.warning(f"Pre-trained model not available: {e}")
                    logger.info("Creating new UME model instead...")
                    # Extract model size from name (e.g., "ume-mini-base-12M" -> "UME_mini")
                    if "mini" in model_name:
                        model_size = "UME_mini"
                    elif "small" in model_name:
                        model_size = "UME_small"
                    elif "medium" in model_name:
                        model_size = "UME_medium"
                    elif "large" in model_name:
                        model_size = "UME_large"
                    else:
                        model_size = "UME_mini"  # Default fallback

                    logger.info(f"Creating new UME model with size: {model_size}")
                    model = UME(
                        model_name=model_size,
                        use_flash_attn=use_flash_attn,
                    )
                    logger.info("New UME model created successfully")
            else:
                # Load from checkpoint path
                logger.info(f"Loading UME model from checkpoint: {model_name}")
                logger.info("Calling UME.load_from_checkpoint...")
                model = UME.load_from_checkpoint(
                    model_name,
                    device=device,
                    use_flash_attn=use_flash_attn,
                )
                logger.info("UME.load_from_checkpoint completed successfully")
        except Exception as e:
            logger.error(f"Failed to load UME model {model_name}: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        logger.info("Freezing model for evaluation...")
        # Freeze model for evaluation
        model.freeze()
        logger.info("Model frozen successfully")

        # Move to specified device (model is already loaded on the correct device)
        if device == "cuda" and torch.cuda.is_available():
            logger.info("Moving model to CUDA...")
            model = model.cuda()
            # Log GPU memory usage if available
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"Using GPU with {gpu_memory:.1f}GB memory")
            except Exception:
                pass
        else:
            logger.info("Model will run on CPU")

        logger.info("Setting up model config...")
        # DGEB expects the returned model to have a config attribute directly
        # UME models have config under model.config, so we need to expose it
        if not hasattr(model, "config"):
            model.config = model.model.config
        logger.info("Model config setup completed")

        logger.info("Storing model as instance variable...")
        # Store model as instance variable
        self.model = model
        logger.info("Model stored successfully")

        # Log final configuration
        flash_status = "enabled" if use_flash_attn else "disabled"
        logger.info(f"Loaded UME model {model_name} on {device} (flash attention: {flash_status})")

        # Return the model for DGEB compatibility
        return model

    def _get_tokenizer(self, model_name: str):
        """Override tokenizer loading to use UME's real tokenizer."""
        # model_name is not used since UME has its own tokenizers built-in
        _ = model_name  # Suppress unused parameter warning

        # Get the appropriate UME tokenizer for the current modality
        lobster_modality = self._get_lobster_modality()

        try:
            # Get the tokenizer transform for the specific modality
            tokenizer_transform = self.model.tokenizer_transforms[lobster_modality]

            # Return the underlying HuggingFace tokenizer
            # This provides the interface that DGEB expects (encode, decode, etc.)
            ume_tokenizer = tokenizer_transform.tokenizer

            logger.debug(
                f"Using UME {lobster_modality.value} tokenizer with vocab size: {len(ume_tokenizer.get_vocab())}"
            )

            return ume_tokenizer

        except Exception as e:
            logger.warning(f"Failed to get UME tokenizer for {lobster_modality.value}: {e}")
            logger.warning("Falling back to dummy tokenizer")

            # Fallback to dummy tokenizer if UME tokenizer access fails
            class DummyTokenizer:
                def __init__(self):
                    self.model_max_length = self.max_seq_length
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
                zero_embeddings = np.zeros((batch_size, len(self.layers), self.embed_dim))
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
        # Get the tokenizer transform for the specified modality
        tokenizer_transform = self.model.tokenizer_transforms[modality]
        encoded_batch = tokenizer_transform(sequences)
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Handle shape (batch, seq) or (batch, 1, seq)
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

        # Get embedding layer and encoder layers
        flexbert = self.model.model.model
        embeddings = flexbert.embeddings(input_ids)
        hidden_states = embeddings
        all_hidden_states = [hidden_states]

        # Use the encoder's forward method which handles unpadding automatically
        # This ensures proper handling of cu_seqlens and max_seqlen for unpadded layers
        _encoder_output = flexbert.encoder(hidden_states=hidden_states, attention_mask=attention_mask)

        # For layer extraction, we need to manually call each layer
        # but we need to handle the unpadding properly
        current_hidden = hidden_states
        for layer in flexbert.encoder.layers:
            # Check if this is an unpadded layer that requires cu_seqlens and max_seqlen
            if hasattr(layer, "forward") and "cu_seqlens" in layer.forward.__code__.co_varnames:
                # This is an unpadded layer, we need to unpad the input
                attention_mask_bool = attention_mask.bool()
                batch, seqlen = current_hidden.shape[:2]
                unpad_hidden, indices, cu_seqlens, max_seqlen = unpad_input(current_hidden, attention_mask_bool)

                # Call the layer with unpadded inputs
                layer_output = layer(
                    unpad_hidden,
                    cu_seqlens,
                    max_seqlen,
                    indices,
                    attn_mask=attention_mask,
                )

                # Repad the output
                current_hidden = pad_input(layer_output, indices, batch, seqlen)
            else:
                # This is a padded layer, call normally
                current_hidden = layer(current_hidden, attn_mask=attention_mask)

            all_hidden_states.append(current_hidden)

        # Select the requested layers (self.layers)
        selected_layers = [
            all_hidden_states[i + 1] for i in self.layers
        ]  # +1 because all_hidden_states[0] is embedding

        # Pool each layer
        pooled_layers = []
        for layer_hidden in selected_layers:
            if self.pool_type == "mean":
                pooled = (layer_hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                    dim=1, keepdim=True
                )
            elif self.pool_type == "max":
                mask = attention_mask.unsqueeze(-1).expand_as(layer_hidden)
                layer_hidden_masked = layer_hidden.masked_fill(mask == 0, float("-inf"))
                pooled = layer_hidden_masked.max(dim=1)[0]
            elif self.pool_type == "cls":
                pooled = layer_hidden[:, 0, :]
            elif self.pool_type == "last":
                lengths = attention_mask.sum(dim=1) - 1
                pooled = torch.stack([layer_hidden[i, l, :] for i, l in enumerate(lengths)], dim=0)
            else:
                raise ValueError(f"Unsupported pool_type: {self.pool_type}")
            pooled_layers.append(pooled)

        # Stack: (batch, num_layers, dim)
        layer_embeddings = torch.stack(pooled_layers, dim=1)

        return layer_embeddings.cpu().numpy()

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
        # Determine current device
        device = next(self.model.parameters()).device.type

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
            "device": device,
            "use_flash_attn": self._use_flash_attn,
            "cuda_available": torch.cuda.is_available(),
        }
