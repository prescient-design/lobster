import logging
import os
import warnings
from collections.abc import Sequence
from typing import Literal
import torch
from torch import Tensor
from torch.nn import Module

from lobster.constants import (
    Modality,
    ModalityType,
    SchedulerType,
)
from lobster.tokenization import UMETokenizerTransform

from .._utils_checkpoint import get_ume_checkpoints, load_checkpoint_with_retry, get_s3_last_modified_timestamp
from ..modern_bert import FlexBERT

warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics.text.perplexity")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class UME(Module):
    """Universal Molecular Encoder.

    A PyTorch module wrapping FlexBert model with useful high-level functions
    for molecular encoding across different modalities.

    Parameters
    ----------
    model_name : Literal["mini", "small", "medium", "UME_large"],
        default="UME_mini"
        Name of the model to initialize.
    max_length : int, default=512
        Maximum sequence length for tokenization.
    lr : float, default=1e-3
        Learning rate for optimizer (used by Lightning wrapper).
    beta1 : float, default=0.9
        Beta1 parameter for Adam optimizer (used by Lightning wrapper).
    beta2 : float, default=0.98
        Beta2 parameter for Adam optimizer (used by Lightning wrapper).
    eps : float, default=1e-12
        Epsilon parameter for Adam optimizer (used by Lightning wrapper).
    mask_percentage : float, default=0.25
        Percentage of tokens to mask during training (used by Lightning wrapper).
    scheduler : str, default="constant_with_warmup"
        Type of learning rate scheduler to use (used by Lightning wrapper).
    num_training_steps : int | None, default=None
        Total number of training steps (used by Lightning wrapper).
    num_warmup_steps : int | None, default=1_000
        Number of warmup steps for learning rate scheduler (used by Lightning wrapper).
    model_kwargs : dict | None, default=None
        Additional keyword arguments to pass to the FlexBERT model.
    scheduler_kwargs : dict | None, default=None
        Additional keyword arguments to pass to the learning rate scheduler (used by Lightning wrapper).
    use_flash_attn : bool, default=True
        Whether to use flash-attn for attention computation. If False, will use standard attention.
        This is useful for CPU-only operation where flash-attn is not available.
    ckpt_path : str | None, default=None
        Path to a checkpoint file to load. Unused.
    weight_decay : float, default=0.0
        Weight decay for optimizer (used by Lightning wrapper).

    Attributes
    ----------
    model : FlexBERT
        The underlying FlexBERT model for encoding.
    tokenizer_transforms : dict[Modality, UMETokenizerTransform]
        Dictionary mapping modality enums to their respective
        tokenizer transforms.
    embedding_dim : int
        Dimension of the output embeddings.
    frozen : bool
        Indicates whether model parameters are frozen.

    Examples
    --------
    >>> # Initialize a new model
    >>> encoder = UME(model_name="UME_mini", max_length=256)
    >>>
    >>> # Load a pretrained model using the convenient from_pretrained method
    >>> encoder = UME.from_pretrained("ume-mini")
    >>>
    >>> # Get embeddings for protein sequences
    >>> sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"]
    >>> embeddings = encoder.embed_sequences(sequences, "amino_acid")
    >>> print(embeddings.shape)
    torch.Size([1, 768])
    """

    def __init__(
        self,
        model_name: Literal["mini", "small", "medium", "large"] = "mini",
        max_length: int = 8192,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        mask_percentage: float = 0.25,
        scheduler: SchedulerType = "constant_with_warmup",
        num_training_steps: int | None = None,
        num_warmup_steps: int | None = 1_000,
        model_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
        use_flash_attn: bool = True,
        ckpt_path: str | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the Universal Molecular Encoder"""
        super().__init__()

        # Instantiate tokenizer transforms for each modality
        self.tokenizer_transforms = {
            modality: UMETokenizerTransform(modality, max_length=max_length, return_modality=True)
            for modality in [Modality.AMINO_ACID, Modality.SMILES, Modality.NUCLEOTIDE]
        }
        # Get any tokenizer to get the special tokens
        tokenizer = list(self.tokenizer_transforms.values())[0].tokenizer

        # Configure model kwargs using helper method
        model_kwargs = self._configure_model_kwargs(model_kwargs, use_flash_attn, ckpt_path)

        # Instantiate the model
        self.model = FlexBERT(
            model_name=f"UME_{model_name}",
            max_length=max_length,
            vocab_size=len(self.get_vocab()),
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            mask_percentage=mask_percentage,
            scheduler=scheduler,
            model_kwargs=model_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.mask_token_id,
            cls_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        self.max_length = max_length
        self.embedding_dim = self.model.config.hidden_size
        self.frozen = False
        self.use_flash_attn = use_flash_attn

    def _configure_model_kwargs(self, model_kwargs: dict | None, use_flash_attn: bool, ckpt_path: str | None) -> dict:
        """Configure model kwargs based on flash attention and checkpoint settings.

        Parameters
        ----------
        model_kwargs : dict | None
            Base model kwargs to extend.
        use_flash_attn : bool
            Whether to use flash attention.
        ckpt_path : str | None
            Path to checkpoint file if loading from checkpoint.

        Returns
        -------
        dict
            Configured model kwargs.
        """
        model_kwargs = model_kwargs or {}
        model_kwargs["use_fa2"] = use_flash_attn

        # Important: If loading from checkpoint, preserve the original architecture
        # A checkpoint trained with flash attention has unpadded layers that can't be changed
        # We can still disable flash attention at the layer level while keeping unpadded architecture
        if ckpt_path is not None:
            # Always use unpadded architecture when loading from checkpoint
            # The individual attention layers will respect the use_fa2 setting
            model_kwargs["padding"] = "unpadded"
            if not use_flash_attn:
                model_kwargs["use_sdpa_attn_mask"] = True
        else:
            # When creating a new model, choose the appropriate architecture
            if use_flash_attn:
                # Flash attention works with unpadded architecture
                model_kwargs["padding"] = "unpadded"
            else:
                # SDPA requires padded architecture to work correctly
                model_kwargs["padding"] = "padded"
                model_kwargs["use_sdpa_attn_mask"] = True

        return model_kwargs

    def get_vocab(self) -> dict[int, str]:
        """Get a consolidated vocabulary from all tokenizers.

        Returns
        -------
        dict[int, str]
            A dictionary mapping token IDs to token strings, sorted by token ID.
            Reserved tokens are excluded.
            Important! Tokens are not unique across modalities and may overlap.
            If the vocabulary is reversed where token strings are keys,
            information will be lost. Use with caution.

        Examples
        --------
        >>> encoder = UME(model_name="UME_mini")
        >>> vocab = encoder.get_vocab()
        >>> print(len(vocab))  # Size of vocabulary
        1536  # Example size
        """
        tokenizers = [transform.tokenizer for transform in self.tokenizer_transforms.values()]

        vocab = {
            token_id: token
            for tokenizer in tokenizers
            for token, token_id in tokenizer.get_vocab().items()
            if "reserved" not in token
        }

        return dict(sorted(vocab.items(), key=lambda item: item[0]))

    def freeze(self) -> None:
        """Freeze the model parameters.

        This method sets requires_grad=False for all model parameters
        and puts the model in evaluation mode.

        Examples
        --------
        >>> encoder = UME(model_name="UME_mini")
        >>> # Check if model is trainable
        >>> print(f"Before freezing - Parameter grad enabled: {next(encoder.model.parameters()).requires_grad}")
        Before freezing - Parameter grad enabled: True
        >>>
        >>> # Freeze the model
        >>> encoder.freeze()
        >>> print(f"Model is frozen: {encoder.frozen}")
        Model is frozen: True
        >>>
        >>> # Now you can use it for inference without gradient computation
        >>> import torch
        >>> embeddings = encoder.embed_sequences(["ACDEFGHIK"], "amino_acid")
        """
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the model parameters.

        This method sets requires_grad=True for all model parameters
        and puts the model in training mode.

        Examples
        --------
        >>> encoder = UME(model_name="UME_mini")
        >>> # First freeze the model
        >>> encoder.freeze()
        >>> print(f"Model is frozen: {encoder.frozen}")
        Model is frozen: True
        >>>
        >>> # Now unfreeze it
        >>> encoder.unfreeze()
        """
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        self.frozen = False

    def embed(
        self,
        inputs: dict[str, Tensor],
        aggregate: bool = True,
    ) -> Tensor:
        """Get embeddings for encoded inputs.

        Parameters
        ----------
        inputs : dict[str, Tensor]
            Dictionary of encoded inputs. Must contain 'input_ids' and 'attention_mask'.
        aggregate : bool, default=True
            Whether to average pool over the sequence length dimension.

        Returns
        -------
        Tensor
            Tensor of embeddings. If aggregate=True, shape is (batch_size, hidden_size).
            Otherwise, shape is (batch_size, seq_len, hidden_size).
        """
        if not all(k in inputs for k in {"input_ids", "attention_mask"}):
            raise ValueError("Missing required keys in inputs: 'input_ids' or 'attention_mask'")

        x = {k: v.to(self.model.device) for k, v in inputs.items() if isinstance(v, Tensor)}

        # Ensure input_ids and attention_mask are 3D (batch_size, 1, length)
        for key in ["input_ids", "attention_mask"]:
            if x[key].dim() == 2:
                x[key] = x[key].unsqueeze(1)

        assert x["input_ids"].ndim == 3
        assert x["input_ids"].shape[1] == 1, (
            f"Input IDs must have shape (batch_size, 1, length), got {x['input_ids'].shape}"
        )

        if self.frozen:
            with torch.no_grad():
                embeddings = self.model.tokens_to_latents(**x)
        else:
            embeddings = self.model.tokens_to_latents(x["input_ids"], x["attention_mask"])

        # Reshape to (batch_size, seq_len, hidden_size)
        batch_size = x["input_ids"].size(0)
        seq_len = x["input_ids"].size(-1)

        if self.model.config.padding == "unpadded":
            embeddings = embeddings.view(batch_size, seq_len, -1)

        # Handle attention mask for both aggregated and token-level outputs
        attention_mask = x["attention_mask"]
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.squeeze(1)  # Remove middle dimension: (batch, seq_len)

        # Ensure attention mask matches embedding sequence length
        if attention_mask.shape[1] != embeddings.shape[1]:
            # Truncate or pad attention mask to match embeddings
            if attention_mask.shape[1] > embeddings.shape[1]:
                attention_mask = attention_mask[:, : embeddings.shape[1]]
            else:
                pad_length = embeddings.shape[1] - attention_mask.shape[1]
                padding = torch.zeros(
                    attention_mask.shape[0], pad_length, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, padding], dim=1)

        # Convert to float and add dimension for broadcasting: (batch, seq_len, 1)
        mask = attention_mask.to(dtype=embeddings.dtype).unsqueeze(-1)

        if aggregate:
            # Apply mask and compute mean only over actual tokens
            masked_embeddings = embeddings * mask
            sum_embeddings = masked_embeddings.sum(dim=1)  # (batch, hidden_dim)
            token_counts = mask.sum(dim=1)  # (batch, 1)

            # Avoid division by zero for empty sequences
            token_counts = torch.clamp(token_counts, min=1.0)
            embeddings = sum_embeddings / token_counts
        else:
            # For token-level embeddings, zero out padding token positions
            # This ensures consistency between flash attention and non-flash attention models
            embeddings = embeddings * mask

        return embeddings

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass for ONNX export and direct model inference.

        This method provides a standard PyTorch forward interface that can be used
        for ONNX export and direct model calls. It wraps the embed method for
        compatibility with standard PyTorch workflows.

        Parameters
        ----------
        input_ids : Tensor
            Input token IDs of shape (batch_size, 1, sequence_length) or (batch_size, sequence_length)
        attention_mask : Tensor
            Attention mask of shape (batch_size, 1, sequence_length) or (batch_size, sequence_length)

        Returns
        -------
        Tensor
            Embeddings of shape (batch_size, hidden_size) with mean pooling applied
        """
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        return self.embed(inputs, aggregate=True)

    def embed_sequences(
        self, sequences: Sequence[str] | str, modality: ModalityType | Modality, aggregate: bool = True
    ) -> Tensor:
        """Embed sequences using the specified modality.

        Parameters
        ----------
        sequences : Sequence[str] | str
            Input sequences to embed.
        modality : ModalityType | Modality
            Modality of the input sequences.
        aggregate : bool, default=True
            Whether to aggregate the embeddings (mean pooling).

        Returns
        -------
        Tensor
            Embeddings of the input sequences.
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        # Get the tokenizer transform for the specified modality
        tokenizer_transform = self.tokenizer_transforms[modality]

        # Tokenize the sequences
        encoded_batch = tokenizer_transform(sequences)

        # Get input_ids and attention_mask
        input_ids = encoded_batch["input_ids"]
        attention_mask = encoded_batch["attention_mask"]

        # Move tensors to the same device as the model
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Fallback for testing or when model has no parameters
            device = getattr(self.model, "device", torch.device("cpu"))
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Create inputs dictionary
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Get embeddings
        embeddings = self.embed(inputs, aggregate=aggregate)

        return embeddings

    def compute_pseudo_likelihood(self, sequences: list[str], modality: Modality) -> list[float]:
        """
        Compute pseudo-likelihood for a batch of sequences.

        Parameters:
        -----------
        sequences : List[str]
            List of sequences to evaluate
        modality : Modality
            The modality of the sequences

        Returns:
        --------
        List[float]
            List of pseudo-likelihood scores for each sequence
        """
        with torch.no_grad():
            try:
                # Filter out empty sequences
                valid_sequences = [seq for seq in sequences if seq.strip()]
                if not valid_sequences:
                    return [0.0] * len(sequences)

                logger.debug(f"Processing {len(valid_sequences)} sequences for modality {modality.value}")

                # Tokenize the sequences using the appropriate tokenizer
                tokenizer_transform = self.tokenizer_transforms[modality]
                logger.debug(f"Using tokenizer transform: {type(tokenizer_transform)}")

                encoded_batch = tokenizer_transform(valid_sequences)
                logger.debug(f"Encoded batch keys: {list(encoded_batch.keys())}")

                # Get input_ids and attention_mask
                input_ids = encoded_batch["input_ids"]  # Shape: (batch_size, 1, seq_len) or (batch_size, seq_len)
                attention_mask = encoded_batch[
                    "attention_mask"
                ]  # Shape: (batch_size, 1, seq_len) or (batch_size, seq_len)

                # Move tensors to the same device as the model
                device = next(self.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Debug: print shapes to understand the actual dimensions
                logger.debug(f"input_ids shape: {input_ids.shape}")
                logger.debug(f"attention_mask shape: {attention_mask.shape}")
                logger.debug(f"Device: {device}")

                # Handle different possible shapes
                if input_ids.dim() == 3:
                    # Shape is (batch_size, 1, seq_len)
                    batch_size, _, seq_len = input_ids.shape
                    input_ids_3d = input_ids  # Already in correct format
                    attention_mask_3d = attention_mask
                elif input_ids.dim() == 2:
                    # Shape is (batch_size, seq_len) - need to add middle dimension
                    batch_size, seq_len = input_ids.shape
                    input_ids_3d = input_ids.unsqueeze(1)  # Add middle dimension: (batch_size, 1, seq_len)
                    attention_mask_3d = attention_mask.unsqueeze(1)
                else:
                    raise ValueError(f"Unexpected input_ids shape: {input_ids.shape}")

                logger.debug(f"After shape handling - batch_size: {batch_size}, seq_len: {seq_len}")
                logger.debug(f"input_ids_3d shape: {input_ids_3d.shape}")

                # Prepare inputs for the model (similar to _compute_mlm_loss)
                # _prepare_inputs expects (batch_size, 1, seq_len) format
                input_ids_flat, attention_mask_flat, cu_seqlens = self.model._prepare_inputs(
                    input_ids_3d, attention_mask_3d
                )

                logger.debug(f"After _prepare_inputs - input_ids_flat shape: {input_ids_flat.shape}")

                # Get model outputs without masking (we want the full sequence)
                hidden_states = self.model.model(
                    input_ids=input_ids_flat,
                    attention_mask=attention_mask_flat,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=self.max_length,
                )

                logger.debug(f"Hidden states shape: {hidden_states.shape}")

                # Get logits from decoder
                logits = self.model.decoder(hidden_states)
                logits = logits.view(-1, self.model.config.vocab_size)  # (batch_size * seq_len, vocab_size)

                logger.debug(f"Logits shape: {logits.shape}")

                # Reshape input_ids for probability computation
                input_ids_reshaped = input_ids_flat.view(-1)  # (batch_size * seq_len)

                # Convert to log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs[torch.arange(len(input_ids_reshaped)), input_ids_reshaped]

                # Reshape back to (batch_size, seq_len)
                token_log_probs = token_log_probs.view(batch_size, seq_len)

                # Average over sequence length to get per-sequence pseudo-likelihood
                # Exclude padding tokens (token_id == self.model.pad_token_id)
                mask = input_ids_3d[:, 0, :] != self.model.pad_token_id
                masked_log_probs = token_log_probs * mask.float()

                # Compute average log probability per sequence
                pseudo_likelihoods = masked_log_probs.sum(dim=1) / (mask.float().sum(dim=1) + 1e-8)

                logger.debug(f"Computed pseudo-likelihoods: {pseudo_likelihoods.shape}")

                # Handle case where some sequences were filtered out
                if len(valid_sequences) < len(sequences):
                    result = [0.0] * len(sequences)
                    valid_idx = 0
                    for i, seq in enumerate(sequences):
                        if seq.strip():
                            result[i] = float(pseudo_likelihoods[valid_idx].cpu().numpy())
                            valid_idx += 1
                    return result

                return pseudo_likelihoods.cpu().numpy().tolist()

            except Exception as e:
                logger.error(f"Error computing pseudo-likelihood: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                # Return zero rewards for failed computations
                return [0.0] * len(sequences)

    def export_onnx(
        self,
        export_path: str,
        modality: ModalityType | Modality,
        sample_sequences: list[str] | None = None,
        opset_version: int = 17,
        device: torch.device | str | None = None,
        dynamic_axes: dict | None = None,
        **export_kwargs,
    ) -> None:
        """Export the UME model to ONNX format.

        This method exports the model using sample sequences to create proper dummy inputs
        that match the expected tokenization format for the specified modality.

        Parameters
        ----------
        export_path : str
            Path to save the ONNX model.
        modality : ModalityType | Modality
            Modality to use for creating dummy inputs. This ensures the ONNX model
            is exported with inputs that match the tokenization format for this modality.
        sample_sequences : list[str] | None, optional
            Sample sequences to use for creating dummy inputs. If None, uses default
            sequences for the specified modality.
        opset_version : int, default=17
            ONNX opset version to use for export.
        device : torch.device | str | None, optional
            Device for dummy inputs. If None, uses the model's device.
        dynamic_axes : dict | None, optional
            Dynamic axes configuration for ONNX export. If None, uses default configuration.
        **export_kwargs
            Additional keyword arguments to pass to torch.onnx.export.

        Examples
        --------
        >>> from lobster.model import UME
        >>> from lobster.constants import Modality
        >>>
        >>> # Initialize model
        >>> ume = UME(model_name="UME_mini")
        >>>
        >>> # Export for SMILES sequences
        >>> ume.export_onnx("ume_smiles.onnx", modality=Modality.SMILES)
        >>>
        >>> # Export for protein sequences with custom samples
        >>> protein_samples = ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"]
        >>> ume.export_onnx("ume_protein.onnx", modality=Modality.AMINO_ACID,
        ...                 sample_sequences=protein_samples)
        """
        device = device or next(self.parameters()).device
        if isinstance(device, str):
            device = torch.device(device)

        # Use default sample sequences if none provided
        if sample_sequences is None:
            sample_sequences = {
                Modality.SMILES: ["CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],  # Shorter SMILES for speed
                Modality.AMINO_ACID: ["MKTVRQERLKSIVRILERSKEPVSGAQL", "ACDEFGHIKL"],  # Protein sequences
                Modality.NUCLEOTIDE: ["ATGCATGC", "GCTAGCTA"],  # DNA sequences
            }[modality]

        # Tokenize the sample sequences to get proper input format
        tokenizer_transform = self.tokenizer_transforms[modality]
        encoded_batch = tokenizer_transform(sample_sequences)

        input_ids = encoded_batch["input_ids"].to(device)
        attention_mask = encoded_batch["attention_mask"].to(device)

        # Ensure 3D format for ONNX export
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1)

        # Ensure correct dtype for ONNX
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        # Use default dynamic axes if none provided
        if dynamic_axes is None:
            dynamic_axes = {
                "input_ids": {0: "batch", 2: "sequence"},
                "attention_mask": {0: "batch", 2: "sequence"},
                "embeddings": {0: "batch"},
            }

        # Export to ONNX
        torch.onnx.export(
            self,
            (input_ids, attention_mask),
            export_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            **export_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: Literal["ume-mini-base-12M", "ume-small-base-90M", "ume-medium-base-480M", "ume-large-base-740M"],
        *,
        device: str | None = None,
        use_flash_attn: bool | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> "UME":
        """Load a pretrained UME model from a model name.

        Currently, we support the following model names:
        - "ume-mini-base-12M"
        - "ume-small-base-90M"
        - "ume-medium-base-480M"
        - "ume-large-base-740M"

        Note: These models are only available to members of Prescient Design for now. Stay
        tuned for UME release.

        Parameters
        ----------
        model_name : str
            Model name from registry.
            Examples:
            - "ume-mini-base-12M" -> loads UME_mini with default checkpoint
        device : str | None, optional
            Device to load the model on ("cpu" or "cuda"). If None, will be determined automatically.
        use_flash_attn : bool | None, optional
            Whether to use flash attention. If None, will be determined based on device.
        cache_dir : str | None, optional
            Directory to cache downloaded models. If None, uses 'models/ume' in current directory.
        **kwargs
            Additional keyword arguments to pass to the model constructor.

        Returns
        -------
        UME
            The loaded pretrained model.

        Examples
        --------
        >>> # Load UME-mini with default checkpoint
        >>> model = UME.from_pretrained("ume-mini-base-12M")
        >>>
        >>> # Load UME-mini with specific device
        >>> model = UME.from_pretrained("ume-mini-base-12M", device="cpu")
        >>>
        >>> # Load with custom cache directory
        >>> model = UME.from_pretrained("ume-mini-base-12M", cache_dir="/path/to/cache")
        """
        import logging

        logger = logging.getLogger(__name__)

        logger.info(f"Loading pretrained UME model: {model_name}")
        logger.debug(f"  - Device: {device}")
        logger.debug(f"  - Use flash attention: {use_flash_attn}")
        logger.debug(f"  - Cache directory: {cache_dir}")

        logger.debug("Fetching checkpoint mapping from S3...")
        checkpoint_dict = get_ume_checkpoints()

        checkpoint_path = checkpoint_dict.get(model_name)

        if checkpoint_path is None:
            available_models = [
                model_name for model_name in checkpoint_dict.keys() if checkpoint_dict[model_name] is not None
            ]
            logger.error(f"Unknown model name: {model_name}")
            logger.error(f"Available models: {available_models}")
            raise ValueError(f"Unknown model name: {model_name}. Currently available models: {available_models}")

        logger.debug(f"Found checkpoint path: {checkpoint_path}")

        # Determine cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "models", "ume")

        logger.debug(f"Using cache directory: {cache_dir}")

        # Get S3 timestamp and include it in the filename
        timestamp = get_s3_last_modified_timestamp(checkpoint_path)
        local_filename = f"{model_name}-{timestamp}.ckpt"
        logger.debug(f"Local filename: {local_filename}")

        # Load the model with automatic retry on corruption
        # happens if previous download was stopped, for example
        logger.debug("Starting model loading with automatic retry...")
        model = load_checkpoint_with_retry(
            checkpoint_path=checkpoint_path,
            local_directory=cache_dir,
            local_filename=local_filename,
            load_func=torch.load,
            device=device,
            use_flash_attn=use_flash_attn,
            **kwargs,
        )

        # Create model instance from checkpoint
        # Extract state dict and model configuration from checkpoint
        if isinstance(model, dict) and "state_dict" in model:
            state_dict = model["state_dict"]
            hyper_parameters = model.get("hyper_parameters", {})
        else:
            raise ValueError("Loaded checkpoint does not contain expected format")

        # Create model instance with checkpoint hyperparameters
        instance = cls(**hyper_parameters, **kwargs)

        # Load state dict
        instance.load_state_dict(state_dict, strict=False)

        # Move to device if specified
        if device is not None:
            instance = instance.to(device)

        # Validate the loaded model
        logger.debug("Validating loaded model configuration...")
        total_params = sum(p.numel() for p in instance.parameters())
        embed_dim = instance.embedding_dim
        num_layers = instance.model.config.num_hidden_layers

        logger.info("Loaded model configuration:")
        logger.info(f"  - Model name: {model_name}")
        logger.info(f"  - Embedding dimension: {embed_dim}")
        logger.info(f"  - Number of layers: {num_layers}")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Parameters in millions: {total_params / 1e6:.1f}M")
        logger.info(f"  - Device: {next(instance.parameters()).device}")
        logger.info(f"  - Use flash attention: {instance.use_flash_attn}")

        # Validate expected model size based on name
        expected_params = {
            "ume-mini-base-12M": (10e6, 20e6),  # 10-20M parameters
            "ume-small-base-90M": (80e6, 100e6),  # 80-100M parameters
            "ume-medium-base-480M": (450e6, 500e6),  # 450-500M parameters
            "ume-large-base-740M": (700e6, 800e6),  # 700-800M parameters
        }

        if model_name in expected_params:
            min_expected, max_expected = expected_params[model_name]
            if min_expected <= total_params <= max_expected:
                logger.debug(
                    f"✅ Model parameter count ({total_params / 1e6:.1f}M) matches expected range for {model_name}"
                )
            else:
                logger.error(
                    f"❌ Model parameter count ({total_params / 1e6:.1f}M) is outside expected range for {model_name} ({min_expected / 1e6:.1f}M-{max_expected / 1e6:.1f}M)"
                )
                raise ValueError(
                    f"Model parameter count mismatch: expected {min_expected / 1e6:.1f}M-{max_expected / 1e6:.1f}M parameters for {model_name}, but got {total_params / 1e6:.1f}M. This indicates a checkpoint mismatch or incorrect model configuration."
                )

        logger.info(f"✅ Successfully loaded pretrained UME model: {model_name}")
        return instance
