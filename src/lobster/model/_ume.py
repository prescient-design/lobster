import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.text import Perplexity

from lobster.constants import Modality, ModalityType
from lobster.tokenization import UmeTokenizerTransform

from .modern_bert import FlexBERT

warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics.text.perplexity")


class Ume(L.LightningModule):
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
    contrastive_loss_weight : float, default=0.0
        Weight for the contrastive loss. Only relevant if the batch contains two inputs.
        Is used to balance the MLM and InfoNCE losses:
        (1 - contrastive_loss_weight) * MLM_loss + contrastive_loss_weight * InfoNCE_loss
        - If contrastive_loss_weight is 0, only MLM is used (default)
        - If contrastive_loss_weight is 1, only InfoNCE is used
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
    tokenizer_transforms : dict[Modality, UmeTokenizerTransform]
        Dictionary mapping modality enums to their respective
        tokenizer transforms.
    embedding_dim : int
        Dimension of the output embeddings.
    frozen : bool
        Indicates whether model parameters are frozen.

    Examples
    --------
    >>> # Initialize a new model
    >>> encoder = Ume(model_name="UME_mini", max_length=256)
    >>>
    >>> # Initialize and load from a checkpoint
    >>> encoder = Ume.load_from_checkpoint("path/to/checkpoint.ckpt")
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
        contrastive_loss_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
        scheduler: Literal[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
            "cosine_with_min_lr",
            "warmup_stable_decay",
        ] = "constant_with_warmup",
        num_training_steps: int | None = None,
        num_warmup_steps: int | None = 1_000,
        model_kwargs: dict | None = None,
        scheduler_kwargs: dict | None = None,
        use_flash_attn: bool = True,
        ckpt_path: str | None = None,
    ) -> None:
        """Initialize the Universal Molecular Encoder"""
        super().__init__()

        self.save_hyperparameters()

        # Instantiate tokenizer transforms for each modality
        self.tokenizer_transforms = {
            modality: UmeTokenizerTransform(modality, max_length=max_length, return_modality=True)
            for modality in Modality
        }

        # Get any tokenizer to get the special tokens
        tokenizer = list(self.tokenizer_transforms.values())[0].tokenizer

        # Prepare model kwargs with flash-attn setting
        model_kwargs = model_kwargs or {}
        model_kwargs["use_fa2"] = use_flash_attn
        if not use_flash_attn:
            model_kwargs["padding"] = "padded"

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
        self.contrastive_loss_weight = contrastive_loss_weight

        # Learnable temperature parameter (like CLIP's logit_scale)
        # Initialize to log(1/temperature) so exp() gives the desired initial temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / contrastive_temperature)))

        # Keep the original for backwards compatibility if needed
        self.contrastive_temperature = contrastive_temperature

        # Metrics need to be attributes so that Lighting will handle moving them to the right device
        for modality in Modality:
            setattr(self, f"train_perplexity/{modality.value}", Perplexity(ignore_index=-100))
            setattr(self, f"val_perplexity/{modality.value}", Perplexity(ignore_index=-100))

    @property
    def current_temperature(self) -> float:
        """Get the current value of the learned temperature parameter.

        Returns
        -------
        float
            Current temperature value (1 / logit_scale.exp())
        """
        return (1.0 / self.logit_scale.exp()).item()

    @property
    def modalities(self) -> list[str]:
        """List of supported modalities.

        Returns
        -------
        list[str]
            The list of supported modality names as strings.

        Examples
        --------
        >>> encoder = Ume(model_name="UME_mini")
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
        >>> encoder = Ume(model_name="UME_mini")
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
        >>> encoder = Ume(model_name="UME_mini")
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
        >>> encoder = Ume(model_name="UME_mini")
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
        >>> encoder = Ume(model_name="UME_mini")
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

    def embed(self, inputs: dict[str, Tensor], aggregate: bool = True) -> Tensor:
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

        if self.frozen:
            with torch.no_grad():
                embeddings = self.model.tokens_to_latents(**x)
        else:
            # Copied from tokens_to_latents to remove the inference_mode decorator, allow training
            if getattr(self.model.config, "padding", "unpadded") == "padded":
                # For padded attention, just pass as (batch, seqlen)
                input_ids = (
                    x["input_ids"].squeeze(1)
                    if x["input_ids"].dim() == 3 and x["input_ids"].shape[1] == 1
                    else x["input_ids"]
                )
                attention_mask = (
                    x["attention_mask"].squeeze(1)
                    if x["attention_mask"].dim() == 3 and x["attention_mask"].shape[1] == 1
                    else x["attention_mask"]
                )
                embeddings = self.model.model(input_ids, attention_mask=attention_mask, max_seqlen=self.max_length)
            else:
                input_ids, attention_mask, cu_seqlens = self.model._prepare_inputs(x["input_ids"], x["attention_mask"])
                embeddings = self.model.model(
                    input_ids, attention_mask=attention_mask, cu_seqlens=cu_seqlens, max_seqlen=self.max_length
                )

        # Reshape to (batch_size, seq_len, hidden_size)
        batch_size = x["input_ids"].size(0)
        seq_len = x["input_ids"].size(-1)
        embeddings = embeddings.view(batch_size, seq_len, -1)

        if aggregate:
            # Simple mean pooling over sequence length dimension
            return embeddings.mean(dim=1)
        else:
            return embeddings

    def embed_sequences(
        self, sequences: Sequence[str] | str, modality: ModalityType | Modality, aggregate: bool = True
    ) -> Tensor:
        """Get embeddings for the provided inputs using the specified modality.

        Parameters
        ----------
        sequences : Sequence[str] | str
            List of input strings to encode or a single string.
        modality : str | Modality
            The modality to use for encoding. Can be a string ("SMILES", "amino_acid",
            "nucleotide", "3d_coordinates") or a Modality enum.
        aggregate : bool, default=True
            Whether to average pool over the sequence length dimension.

        Returns
        -------
        Tensor
            Tensor of embeddings. If aggregate=True, shape is (batch_size, hidden_size).
            Otherwise, shape is (batch_size, seq_len, hidden_size).

        Raises
        ------
        ValueError
            If the model has not been initialized with a checkpoint.

        Examples
        --------
        >>> # Get protein embeddings
        >>> encoder = Ume(model_name="UME_mini")
        >>> sequences = ["MKTVQRERL", "ACDEFGHIKL"]
        >>> embeddings = encoder.embed_sequences(sequences, "amino_acid")
        >>> print(embeddings.shape)
        torch.Size([2, 768])
        >>>
        >>> # Get token-level embeddings for DNA sequences
        >>> dna_seqs = ["ATGCATGC", "GCTAGCTA"]
        >>> token_embeddings = encoder.embed_sequences(dna_seqs, "nucleotide", aggregate=False)
        >>> print(token_embeddings.shape)
        torch.Size([2, 10, 768])  # [batch_size, seq_len, hidden_dim] (includes special tokens)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        modality_enum = Modality(modality) if isinstance(modality, str) else modality
        tokenizer_transform = self.tokenizer_transforms[modality_enum]

        encoded = tokenizer_transform(list(sequences))

        return self.embed(encoded, aggregate=aggregate)

    def configure_optimizers(self) -> dict[str, object]:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        dict[str, object]
            Dictionary containing optimizer and learning rate scheduler.
        """
        return self.model.configure_optimizers()

    def _get_logits_and_labels(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Process inputs with different modalities and get logits and labels for training."""
        # New shape: (batch_size * seq_len)
        input_ids, attention_mask, cu_seqlens = self.model._prepare_inputs(batch["input_ids"], batch["attention_mask"])

        masked_input_ids, labels = self.model._mask_inputs(input_ids)

        hidden_states = self.model.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=self.max_length,
        )

        # Get logits from decoder
        logits = self.model.decoder(hidden_states)

        # Reshape for loss calculation
        logits = logits.view(-1, self.model.config.vocab_size)  # (batch_size * sequence_length, vocab_size)
        labels = labels.view(-1)  # (batch_size * sequence_length)

        return logits, labels

    def _split_combined_batch(self, combined_batch: dict[str, Tensor | list[Modality]]) -> tuple[Tensor, Tensor]:
        """Split a combined batch of two inputs into two separate batches.

        Parameters
        ----------
        combined_batch : dict[str, Tensor | list[Modality]]
            The combined batch to split.

        Returns
        -------
        tuple[dict[str, Tensor | list[Modality]], dict[str, Tensor | list[Modality]]]
            A tuple of two dictionaries, each containing the input ID, attention masks,
            and modality for the two separate batches.
        """
        input_ids_a = combined_batch["input_ids"][:, 0, :].unsqueeze(1).contiguous()
        input_ids_b = combined_batch["input_ids"][:, 1, :].unsqueeze(1).contiguous()
        attention_mask_a = combined_batch["attention_mask"][:, 0, :].unsqueeze(1).contiguous()
        attention_mask_b = combined_batch["attention_mask"][:, 1, :].unsqueeze(1).contiguous()

        modality_list = (
            combined_batch["metadata"]["modality"] if "metadata" in combined_batch else combined_batch["modality"]
        )
        modality_a = [t[0] for t in modality_list]
        modality_b = [t[1] for t in modality_list]

        return (
            {
                "input_ids": input_ids_a,
                "attention_mask": attention_mask_a,
                "modality": modality_a,
            },
            {
                "input_ids": input_ids_b,
                "attention_mask": attention_mask_b,
                "modality": modality_b,
            },
        )

    def _standard_infonce_loss(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        """Compute InfoNCE loss using the standard approach with full similarity matrix.

        Parameters
        ----------
        embeddings_a : Tensor
            First set of normalized embeddings, shape (batch_size, hidden_size)
        embeddings_b : Tensor
            Second set of normalized embeddings, shape (batch_size, hidden_size)

        Returns
        -------
        Tensor
            InfoNCE loss
        """
        # Compute similarity matrix using learnable temperature (like CLIP)
        # logit_scale.exp() converts from log-space back to the actual scale
        similarities = embeddings_a @ embeddings_b.T * self.logit_scale.exp()

        # Create labels (diagonal should be positive)
        labels = torch.arange(embeddings_a.shape[0], device=embeddings_a.device)

        # InfoNCE loss in both directions
        loss_a = nn.functional.cross_entropy(similarities, labels)
        loss_b = nn.functional.cross_entropy(similarities.T, labels)

        return (loss_a + loss_b) / 2

    def _infonce_step(
        self,
        batch_a: dict[str, Tensor | list[Modality]],
        batch_b: dict[str, Tensor | list[Modality]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Perform a contrastive step.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            The batch to perform the contrastive step on.
        stage : Literal["train", "val"]
            The stage to perform the contrastive step on.

        Returns
        -------
        Tensor
            The loss for the contrastive step.
        """
        # Embed both sides separately (could merge for more efficiency but I don't think it matters)
        embeddings_a = self.embed(batch_a, aggregate=True)
        embeddings_b = self.embed(batch_b, aggregate=True)

        assert embeddings_a.shape == embeddings_b.shape
        assert embeddings_a.shape == (batch_a["input_ids"].shape[0], self.model.config.hidden_size)

        # Normalize embeddings once for both implementations
        embeddings_a = nn.functional.normalize(embeddings_a, dim=-1)
        embeddings_b = nn.functional.normalize(embeddings_b, dim=-1)

        loss = self._standard_infonce_loss(embeddings_a, embeddings_b)

        self.log(f"contrastive_{stage}_loss", loss, rank_zero_only=True, sync_dist=True)

        # Log the current learned temperature for monitoring
        current_temperature = 1.0 / self.logit_scale.exp()
        self.log(f"contrastive_{stage}_temperature", current_temperature, rank_zero_only=True, sync_dist=True)

        return loss

    def _mlm_step(self, batch: dict[str, Tensor | list[Modality]], stage: Literal["train", "val"]) -> Tensor:
        """Perform a masked language model step.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            The batch to perform the masked language model step on.
        stage : Literal["train", "val"]
            The stage to perform the masked language model step on.

        Returns
        -------
        Tensor
            The loss for the masked language model step.
        """
        batch_size, length = batch["input_ids"].shape[0], batch["input_ids"].shape[2]

        logits, labels = self._get_logits_and_labels(batch)

        # Compute loss
        loss = self.model.loss_fn(logits, labels)
        self.log(f"mlm_{stage}_loss", loss, rank_zero_only=True, sync_dist=True)

        # Compute overall perplextiy
        perplexity = torch.exp(loss)
        self.log(f"{stage}_perplexity", perplexity, rank_zero_only=True, sync_dist=True)

        # Compute perplexity for each modality separately
        modalities: list[Modality] = batch["metadata"]["modality"] if "metadata" in batch else batch["modality"]

        logits_reshaped = logits.view(batch_size, length, self.model.vocab_size)
        labels_reshaped = labels.view(batch_size, length)

        for modality in set(modalities):
            mask = torch.tensor([m == modality for m in modalities], device=self.device, dtype=torch.bool)

            if not mask.any():
                continue

            metric_name = f"{stage}_perplexity/{modality}"
            metric = getattr(self, metric_name)
            metric(logits_reshaped[mask], labels_reshaped[mask])

            # Do not specify on_step since Lightning will handle this automatically
            self.log(metric_name, metric, rank_zero_only=True, sync_dist=True)

        return loss

    def _delegate_step_by_batch_shape(
        self, batch: dict[str, Tensor | list[Modality]], stage: Literal["train", "val"]
    ) -> Tensor:
        # If the batch contains only one input, we can only run MLM
        if batch["input_ids"].shape[1] == 1:
            return self._mlm_step(batch, stage)

        # If there are two items in the batch, we run MLM on the first item and InfoNCE on the second item
        # (the first item is the original modality and the second item is the converted modality)
        elif batch["input_ids"].shape[1] == 2:
            batch_a, batch_b = self._split_combined_batch(batch)

            # See if we can skip InfoNCE
            if self.contrastive_loss_weight > 0:
                contrastive_loss = self._infonce_step(batch_a, batch_b, stage)
            else:
                contrastive_loss = torch.tensor(0.0, device=self.device)

            # See if we can skip MLM
            if self.contrastive_loss_weight != 1.0:
                mlm_loss = self._mlm_step(batch_a, stage)
            else:
                mlm_loss = torch.tensor(0.0, device=self.device)

            return (1 - self.contrastive_loss_weight) * mlm_loss + self.contrastive_loss_weight * contrastive_loss

        else:
            raise ValueError(f"Not sure what '{stage}' step to run with inputs of shape: {batch['input_ids'].shape}")

    def training_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        loss = self._delegate_step_by_batch_shape(batch, "train")
        self.log("train_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        loss = self._delegate_step_by_batch_shape(batch, "val")
        self.log("val_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss
