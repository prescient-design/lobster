import logging
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import lightning as L
import torch
import transformers
from torch import Tensor
from torchmetrics.text import Perplexity

from lobster.constants import (
    Modality,
    ModalityType,
    SchedulerType,
)
from lobster.tokenization import UMETokenizerTransform

from ._utils_checkpoint import get_ume_checkpoints, load_checkpoint_with_retry
from .losses import InfoNCELoss, SymileLoss
from .modern_bert import FlexBERT

warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics.text.perplexity")

logger = logging.getLogger(__name__)


class UME(L.LightningModule):
    """Universal Molecular Encoder.

    A light wrapper around FlexBert model with useful high-level functions
    for molecular encoding across different modalities.

    Parameters
    ----------
    model_name : Literal["UME_mini", "UME_small", "UME_medium", "UME_large"],
        default="UME_mini"
        Name of the model to initialize.
    max_length : int, default=512
        Maximum sequence length for tokenization.
    lr : float, default=1e-3
        Learning rate for optimizer.
    beta1 : float, default=0.9
        Beta1 parameter for Adam optimizer.
    beta2 : float, default=0.98
        Beta2 parameter for Adam optimizer.
    eps : float, default=1e-12
        Epsilon parameter for Adam optimizer.
    mask_percentage : float, default=0.25
        Percentage of tokens to mask during training.
    contrastive_loss_type : ContrastiveLossType, default=None
        Type of contrastive loss to use. Options:
        - None: Only use MLM loss
        - "symile": Use Symile loss for multiple modality views of the same input (>= 2 views)
        - "clip": Use standard CLIP-style InfoNCE loss (2 views)
        - "disco_clip": Use distributed CLIP loss for memory efficiency (2 views)
    contrastive_loss_weight : float, default=0.0
        Weight for the contrastive loss. Only relevant if contrastive_loss_type is not None.
        Is used to balance the MLM and contrastive losses:
        (1 - contrastive_loss_weight) * MLM_loss + contrastive_loss_weight * contrastive_loss
        - If contrastive_loss_weight is 0, only MLM is used (default)
        - If contrastive_loss_weight is 1, only contrastive loss is used
        - If 0 < contrastive_loss_weight < 1, both are used
    contrastive_temperature : float, default=0.07
        Temperature for the contrastive loss.
    scheduler : str, default="constant_with_warmup"
        Type of learning rate scheduler to use.
    num_training_steps : int | None, default=None
        Total number of training steps.
    num_warmup_steps : int | None, default=1_000
        Number of warmup steps for learning rate scheduler.
    model_kwargs : dict | None, default=None
        Additional keyword arguments to pass to the FlexBERT model.
    scheduler_kwargs : dict | None, default=None
        Additional keyword arguments to pass to the learning rate scheduler.
    use_flash_attn : bool, default=True
        Whether to use flash-attn for attention computation. If False, will use standard attention.
        This is useful for CPU-only operation where flash-attn is not available.
    ckpt_path : str | None, default=None
        Path to a checkpoint file to load. Unused.

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
    >>> # Initialize and load from a checkpoint
    >>> encoder = UME.load_from_checkpoint("path/to/checkpoint.ckpt")
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
        model_name: Literal["UME_mini", "UME_small", "UME_medium", "UME_large"] = "UME_mini",
        max_length: int = 8192,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        mask_percentage: float = 0.25,
        contrastive_loss_type: Literal[None, "symile", "clip", "disco_clip"] = None,
        contrastive_loss_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
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

        self.save_hyperparameters()

        # Instantiate tokenizer transforms for each modality
        self.tokenizer_transforms = {
            modality: UMETokenizerTransform(modality, max_length=max_length, return_modality=True)
            for modality in [Modality.AMINO_ACID, Modality.SMILES, Modality.NUCLEOTIDE]
        }

        # Get any tokenizer to get the special tokens
        tokenizer = list(self.tokenizer_transforms.values())[0].tokenizer

        # Prepare model kwargs with flash-attn setting
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

        # Instantiate the model
        self.model = FlexBERT(
            model_name=model_name,
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
        self.contrastive_loss_type = contrastive_loss_type
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_temperature = contrastive_temperature
        self.use_flash_attn = use_flash_attn
        self._weight_decay = weight_decay
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Initialize loss functions
        self.symile_loss_fn = SymileLoss(logit_scale=1.0 / contrastive_temperature)
        self.infonce_loss_fn = InfoNCELoss(
            temperature=contrastive_temperature,
            use_disco=contrastive_loss_type == "disco_clip",
        )

        # Metrics need to be attributes so that Lighting will handle moving them to the right device
        for modality in Modality:
            setattr(self, f"train_perplexity/{modality.value}", Perplexity(ignore_index=-100))
            setattr(self, f"val_perplexity/{modality.value}", Perplexity(ignore_index=-100))

    @property
    def modalities(self) -> list[str]:
        """List of supported modalities.

        Returns
        -------
        list[str]
            The list of supported modality names as strings.

        Examples
        --------
        >>> encoder = UME(model_name="UME_mini")
        >>> print(encoder.modalities)
        ['SMILES', 'amino_acid', 'nucleotide', '3d_coordinates']
        """
        return [modality.value for modality in Modality]

    def get_tokenizer(self, modality: ModalityType | Modality) -> Callable:
        """Get the appropriate tokenizer for the given modality.

        Parameters
        ----------
        modality : str | Modality
            The modality to use for encoding. Can be a string ("SMILES", "amino_acid",
            "nucleotide", "3d_coordinates") or a Modality enum.

        Returns
        -------
        Callable
            The appropriate tokenizer for the specified modality.

        Examples
        --------
        >>> encoder = UME(model_name="UME_mini")
        >>>
        >>> # Get tokenizer for amino acid sequences
        >>> tokenizer = encoder.get_tokenizer("amino_acid")
        >>> sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQL"]
        >>> tokens = tokenizer(sequences, return_tensors="pt")
        >>> print(tokens.keys())
        dict_keys(['input_ids', 'attention_mask'])
        >>>
        >>> # Get tokenizer for nucleotide sequences using Modality enum
        >>> from lobster.constants import Modality
        >>> dna_tokenizer = encoder.get_tokenizer(Modality.NUCLEOTIDE)
        >>> dna_sequences = ["ATGCATTGCA"]
        >>> dna_tokens = dna_tokenizer(dna_sequences, return_tensors="pt")
        >>> print(dna_tokens["input_ids"].shape)
        torch.Size([1, 12])  # Including special tokens
        >>>
        >>> # Process SMILES strings with tokenizer
        >>> tokenizer = encoder.get_tokenizer("SMILES")
        >>> smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]  # Aspirin
        >>> tokens = tokenizer(smiles, return_tensors="pt")
        >>> print(tokens["attention_mask"].sum())  # Number of non-padding tokens
        tensor(23)
        """
        modality_enum = Modality(modality) if isinstance(modality, str) else modality

        return self.tokenizer_transforms[modality_enum].tokenizer

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

    def _extract_batch_components(
        self,
        batch: dict[str, Tensor | list[Modality]],
        index: int,
    ) -> dict[str, Tensor | list[Modality]]:
        """Extract components for a specific view from a combined batch."""
        input_ids = batch["input_ids"][:, index, :].unsqueeze(1).contiguous()
        attention_mask = batch["attention_mask"][:, index, :].unsqueeze(1).contiguous()

        modality_list = batch["metadata"]["modality"] if "metadata" in batch else batch["modality"]
        modality = [t[index] for t in modality_list]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "modality": modality,
        }

    def _split_combined_batch(
        self,
        combined_batch: dict[str, Tensor | list[Modality]],
    ) -> tuple[dict[str, Tensor | list[Modality]], ...]:
        """Split a combined batch of N inputs into N separate batches."""
        num_splits = combined_batch["input_ids"].shape[1]

        return tuple(self._extract_batch_components(combined_batch, i) for i in range(num_splits))

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

        if aggregate:
            # Use mean pooling over sequence length dimension
            embeddings = embeddings.mean(dim=1)

        return embeddings

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
            weight_decay=self._weight_decay,
        )

        scheduler = transformers.get_scheduler(
            self.scheduler,
            optimizer,
            num_training_steps=self._num_training_steps,
            num_warmup_steps=self._num_warmup_steps,
            scheduler_specific_kwargs=self.scheduler_kwargs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _compute_infonce_loss(
        self,
        embeddings: list[Tensor],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute contrastive loss between two batches."""
        if len(embeddings) != 2:
            raise ValueError(
                f"{self.__class__.__name__} with InfoNCE loss requires exactly 2 views, got {len(embeddings)}"
            )

        embeddings_a, embeddings_b = embeddings
        assert embeddings_a.shape == embeddings_b.shape

        loss = self.infonce_loss_fn(embeddings_a, embeddings_b)

        return loss

    def _infonce_step(
        self,
        batch_a: dict[str, Tensor | list[Modality]],
        batch_b: dict[str, Tensor | list[Modality]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Perform a contrastive step with optional MLM mixing."""
        # Compute embeddings for both batches
        embeddings = [self.embed(batch) for batch in [batch_a, batch_b]]

        contrastive_loss = (
            self._compute_infonce_loss(embeddings, stage)
            if self.contrastive_loss_weight > 0
            else torch.tensor(0.0, device=self.device)
        )

        mlm_loss = (
            self._compute_mlm_loss(batch_a, stage)
            if self.contrastive_loss_weight != 1.0
            else torch.tensor(0.0, device=self.device)
        )

        return self._compute_weighted_loss(
            contrastive_loss=contrastive_loss,
            mlm_loss=mlm_loss,
            stage=stage,
        )

    def _process_batch_for_modality_metrics(
        self,
        logits: Tensor,
        labels: Tensor,
        modalities: list[Modality],
        stage: Literal["train", "val"],
    ) -> None:
        """Process batch to compute per-modality metrics."""
        # Calculate batch size from modalities length
        batch_size = len(modalities)
        seq_length = logits.shape[0] // batch_size

        # Reshape logits and labels to (batch_size, seq_length, vocab_size) and (batch_size, seq_length)
        logits_reshaped = logits.view(batch_size, seq_length, -1)
        labels_reshaped = labels.view(batch_size, seq_length)

        for modality in set(modalities):
            mask = torch.tensor([m == modality for m in modalities], device=self.device, dtype=torch.bool)
            if not mask.any():
                continue

            metric_name = f"{stage}_perplexity/{modality}"

            if not hasattr(self, metric_name):
                logger.warning(f"Metric {metric_name} not found in {self.__class__.__name__}. Skipping.")
                continue

            metric = getattr(self, metric_name)
            metric(logits_reshaped[mask], labels_reshaped[mask])

            self.log(metric_name, metric, rank_zero_only=True, sync_dist=True)

    def _compute_weighted_loss(
        self,
        mlm_loss: Tensor,
        contrastive_loss: Tensor,
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute weighted loss combining MLM and contrastive losses."""
        # Log individual losses
        self.log(f"mlm_{stage}_loss", mlm_loss, rank_zero_only=True, sync_dist=True)
        self.log(f"contrastive_{stage}_loss", contrastive_loss, rank_zero_only=True, sync_dist=True)

        # Compute weighted loss
        total_loss = (1 - self.contrastive_loss_weight) * mlm_loss + self.contrastive_loss_weight * contrastive_loss
        self.log(f"{stage}_loss", total_loss, rank_zero_only=True, sync_dist=True)

        return total_loss

    def _compute_mlm_loss(
        self,
        batch: dict[str, Tensor | list[Modality]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute masked language model loss."""
        # Prepare inputs for the model
        input_ids, attention_mask, cu_seqlens = self.model._prepare_inputs(batch["input_ids"], batch["attention_mask"])
        masked_input_ids, labels = self.model._mask_inputs(input_ids)

        # Get model outputs
        hidden_states = self.model.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=self.max_length,
        )

        # Get logits from decoder and reshape for loss calculation
        logits = self.model.decoder(hidden_states)
        logits = logits.view(-1, self.model.config.vocab_size)  # (batch_size * sequence_length, vocab_size)
        labels = labels.view(-1)  # (batch_size * sequence_length)

        # Compute loss
        loss = self.model.loss_fn(logits, labels)

        # Log overall metrics
        perplexity = torch.exp(loss)
        self.log(f"{stage}_perplexity", perplexity, rank_zero_only=True, sync_dist=True)

        # Process per-modality metrics
        modalities = batch["metadata"]["modality"] if "metadata" in batch else batch["modality"]
        self._process_batch_for_modality_metrics(logits, labels, modalities, stage)

        return loss

    def _compute_symile_loss(
        self,
        embeddings: list[Tensor],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute Symile loss for a batch of N views of the same entity."""
        embeddings = [torch.nn.functional.normalize(embedding, dim=-1) for embedding in embeddings]

        return self.symile_loss_fn(embeddings)

    def _contrastive_step(
        self,
        *batches: dict[str, Tensor | list[Modality]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Perform a contrastive step with optional MLM mixing."""
        if len(batches) < 2:
            raise ValueError(f"Contrastive loss requires at least 2 views but got {len(batches)}: {batches}")

        if self.contrastive_loss_type in ["clip", "disco_clip"]:
            if len(batches) != 2:
                raise ValueError("InfoNCE loss requires exactly 2 views")

        embeddings = [self.embed(batch) for batch in batches]

        contrastive_loss_fn = (
            self._compute_symile_loss if self.contrastive_loss_type == "symile" else self._compute_infonce_loss
        )

        contrastive_loss = (
            contrastive_loss_fn(embeddings, stage=stage)
            if self.contrastive_loss_weight > 0
            else torch.tensor(0.0, device=self.device)
        )

        mlm_loss = (
            self._compute_mlm_loss(batches[0], stage)
            if self.contrastive_loss_weight != 1.0
            else torch.tensor(0.0, device=self.device)
        )

        return self._compute_weighted_loss(
            contrastive_loss=contrastive_loss,
            mlm_loss=mlm_loss,
            stage=stage,
        )

    def _delegate_step_by_batch_shape(
        self,
        batch: dict[str, Tensor | list[Modality]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Delegate to appropriate loss computation based on batch shape."""
        # Validate batch structure
        assert batch["input_ids"].ndim == 3, (
            f"Batch must have shape (batch_size, num_views, sequence_length) but got {batch['input_ids'].shape}"
        )
        assert batch["input_ids"].shape[1] > 0, "Number of views must be positive"

        num_views = batch["input_ids"].shape[1]

        # If no contrastive loss is specified, only use MLM
        if self.contrastive_loss_type is None:
            if num_views > 1:
                raise ValueError(f"Contrastive loss type is None but num_views > 1 ({num_views})")
            return self._compute_mlm_loss(batch, stage)

        batches = self._split_combined_batch(batch)

        return self._contrastive_step(*batches, stage=stage)

    def training_step(
        self,
        batch: dict[str, Tensor | list[Modality]],
        batch_idx: int,
    ) -> Tensor:
        """Perform a single training step.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            Input batch
        batch_idx : int
            Index of the current batch

        Returns
        -------
        Tensor
            Computed loss
        """
        loss = self._delegate_step_by_batch_shape(batch, "train")
        self.log("train_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: dict[str, Tensor | list[Modality]],
        batch_idx: int,
    ) -> Tensor:
        """Perform a single validation step.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            Input batch
        batch_idx : int
            Index of the current batch

        Returns
        -------
        Tensor
            Computed loss
        """
        loss = self._delegate_step_by_batch_shape(batch, "val")
        self.log("val_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        use_flash_attn: bool | None = None,
        device: str | None = None,
        **kwargs,
    ) -> "UME":
        """Load a model from a checkpoint with device-specific configuration.

        This method configures the model based on the specified or available device:
        - For CPU: Uses padded architecture with SDPA attention mask
        - For GPU: Uses unpadded architecture with Flash Attention

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        use_flash_attn : bool | None, optional
            Whether to use flash attention. If None, will be determined based on device.
        device : str | None, optional
            Device to load the model on ("cpu" or "cuda"). If None, will be determined automatically.
        *args
            Additional positional arguments to pass to the parent class's load_from_checkpoint.
        **kwargs
            Additional keyword arguments to pass to the parent class's load_from_checkpoint.

        Returns
        -------
        UME
            The loaded model with appropriate device-specific configuration.

        Raises
        ------
        ValueError
            If an invalid device is specified.
        """
        # Determine device
        if device is not None:
            if device not in ["cpu", "cuda"]:
                raise ValueError(f"Invalid device: {device}. Must be one of ['cpu', 'cuda']")
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA device requested but not available")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine flash attention setting
        if use_flash_attn is None:
            use_flash_attn = device == "cuda"

        # Configure model based on device
        model_kwargs = kwargs.pop("model_kwargs", {})
        model_kwargs.update(
            {
                "use_fa2": use_flash_attn,
                "padding": "unpadded" if use_flash_attn else "padded",
                "use_sdpa_attn_mask": not use_flash_attn,
            }
        )
        kwargs["model_kwargs"] = model_kwargs
        kwargs["use_flash_attn"] = use_flash_attn

        # Load the model using the parent class's method
        model = super().load_from_checkpoint(checkpoint_path, *args, **kwargs)

        # Move model to specified device
        model = model.to(device)

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name: Literal["ume-mini-base-12M", "ume-medium-base-480M", "ume-large-base-740M"],
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
            Additional keyword arguments to pass to load_from_checkpoint.

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

        # Warning that you're using pre-release checkpoints which
        # are just placeholder checkpoints for now.
        warnings.warn(
            "You're using pre-release UME checkpoints which are just placeholder checkpoints for now. Stay tuned for UME release.",
            stacklevel=2,
        )
        checkpoint_dict = get_ume_checkpoints()

        checkpoint_path = checkpoint_dict.get(model_name)
        if checkpoint_path is None:
            available_models = [
                model_name for model_name in checkpoint_dict.keys() if checkpoint_dict[model_name] is not None
            ]
            raise ValueError(f"Unknown model name: {model_name}. Currently available models: {available_models}")

        # Determine cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "models", "ume")

        local_filename = f"{model_name}.ckpt"

        # Load the model with automatic retry on corruption
        # happens if previous download was stopped, for example
        return load_checkpoint_with_retry(
            checkpoint_path=checkpoint_path,
            local_directory=cache_dir,
            local_filename=local_filename,
            load_func=cls.load_from_checkpoint,
            device=device,
            use_flash_attn=use_flash_attn,
            **kwargs,
        )
