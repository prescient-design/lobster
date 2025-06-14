import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from symile import MIPSimilarity, Symile
from torch import Tensor
from torchmetrics.text import Perplexity

from lobster.constants import Modality, ModalityType
from lobster.tokenization import UmeTokenizerTransform

from ._disco_clip import Gather
from ._distributed_utils import get_rank, is_distributed
from .modern_bert import FlexBERT

warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics.text.perplexity")

logger = logging.getLogger(__name__)


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
    use_distributed_clip : bool, default=True
        Whether to use DisCo-CLIP distributed contrastive loss for memory efficiency.
        Disco-CLIP enables gathering embeddings across all GPUs which results in
        more memory-efficient training (because we share only embeddings across not
        full activations). This way we get more negative examples.
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
        use_distributed_clip: bool = True,
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
            for modality in [Modality.AMINO_ACID, Modality.SMILES, Modality.NUCLEOTIDE]
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

        self.contrastive_temperature = contrastive_temperature
        self.use_distributed_clip = use_distributed_clip
        self.symile_loss = Symile()
        self.mip_similarity = MIPSimilarity()

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
                x[key] = x[key].unsqueeze(1).contiguous()

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
        return super().configure_optimizers()

    def _compute_mlm_loss(self, batch: dict[str, Tensor | list[Modality]], stage: Literal["train", "val"]) -> Tensor:
        """Compute the masked language modeling loss for a batch.

        This method processes inputs with different modalities, computes logits and labels,
        and returns the MLM loss along with logging perplexity metrics.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            The batch containing input_ids, attention_mask, and modality information.
        stage : Literal["train", "val"]
            The current stage (training or validation).

        Returns
        -------
        Tensor
            The computed MLM loss.
        """
        batch_size, length = batch["input_ids"].shape[0], batch["input_ids"].shape[1]

        # Reshape batch to (batch_size, 1, seq_len)
        batch = {
            "input_ids": batch["input_ids"].unsqueeze(1).contiguous(),
            "attention_mask": batch["attention_mask"].unsqueeze(1).contiguous(),
            "modality": batch["modality"],
        }

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

    def _compute_infonce_loss(
        self,
        batch_a: dict[str, Tensor | list[Modality]],
        batch_b: dict[str, Tensor | list[Modality]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute the InfoNCE contrastive loss between two batches.

        This method computes the InfoNCE loss between embeddings from two different batches,
        supporting both standard and distributed (DisCo-CLIP) computation.

        Parameters
        ----------
        batch_a : dict[str, Tensor | list[Modality]]
            First batch for contrastive learning.
        batch_b : dict[str, Tensor | list[Modality]]
            Second batch for contrastive learning.
        stage : Literal["train", "val"]
            The current stage (training or validation).

        Returns
        -------
        Tensor
            The computed InfoNCE loss.
        """
        # Embed both sides separately
        embeddings_a = self.embed(batch_a, aggregate=True)
        embeddings_b = self.embed(batch_b, aggregate=True)

        assert embeddings_a.shape == embeddings_b.shape
        assert embeddings_a.shape == (batch_a["input_ids"].shape[0], self.model.config.hidden_size)

        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2.0, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2.0, dim=1)

        # Choose loss computation method based on configuration
        if self.use_distributed_clip and is_distributed():
            # Gather embeddings from all GPUs using DisCo-CLIP
            all_embeddings_a = Gather(embeddings_a)
            all_embeddings_b = Gather(embeddings_b)

            # Get local batch size and rank for label calculation
            local_batch_size = embeddings_a.shape[0]
            rank = get_rank()

            # Compute local similarities using DisCo-CLIP approach with slicing
            logits_a = (
                all_embeddings_a[local_batch_size * rank : local_batch_size * (rank + 1)]
                @ all_embeddings_b.T
                / self.contrastive_temperature
            )
            logits_b = (
                all_embeddings_b[local_batch_size * rank : local_batch_size * (rank + 1)]
                @ all_embeddings_a.T
                / self.contrastive_temperature
            )

            # Create labels - positive pairs are at positions offset by rank * local_batch_size
            labels = torch.arange(local_batch_size, device=self.device) + rank * local_batch_size

        else:
            # Standard InfoNCE loss computation
            logits_a = embeddings_a @ embeddings_b.T / self.contrastive_temperature
            logits_b = logits_a.T
            labels = torch.arange(embeddings_a.shape[0], device=self.device)

        # InfoNCE loss in both directions
        loss_a = nn.functional.cross_entropy(logits_a, labels)
        loss_b = nn.functional.cross_entropy(logits_b, labels)

        loss = (loss_a + loss_b) / 2

        self.log(f"contrastive_{stage}_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss

    def _compute_symile_loss(
        self,
        batches: list[dict[str, Tensor | list[Modality]]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute the Symile contrastive loss for multiple modalities.

        This method computes the Symile loss between embeddings from multiple batches,
        supporting contrastive learning across unlimited modalities.

        Reference
        ----------
        @inproceedings{saporta2024symile,
            title = {Contrasting with Symile: Simple Model-Agnostic Representation Learning for Unlimited Modalities}
            author = {Saporta, Adriel and Puli, Aahlad and Goldstein, Mark and Ranganath, Rajesh}
            booktitle = {Advances in Neural Information Processing Systems},
            year = {2024}
        }

        Parameters
        ----------
        batches : list[dict[str, Tensor | list[Modality]]]
            List of batches for each modality.
        stage : Literal["train", "val"]
            The current stage (training or validation).

        Returns
        -------
        Tensor
            The computed Symile loss.
        """
        embeddings = [self.embed(batch, aggregate=True) for batch in batches]
        embeddings = [F.normalize(emb, p=2.0, dim=1) for emb in embeddings]

        loss = self.symile_loss(embeddings, self.contrastive_temperature)

        self.log(f"symile_{stage}_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss

    def _split_batch_by_index(
        self, batch: dict[str, Tensor | list[Modality]], index: int
    ) -> dict[str, Tensor | list[Modality]]:
        """Split a batch by input index for contrastive learning.

        Two types of batches are supported:
        - Batch with samples with one representation
            e.g. {
                "input_ids": shape (batch_size, 1, seq_len)
                "attention_mask": shape (batch_size, 1, seq_len)
                "modality": ["amino_acid", "nucleotide"]
            }
            Contains two independent samples with one representation each.

        - Batch with samples with multiple representations
            e.g. {
                "input_ids": shape (batch_size, num_modalities, seq_len)
                "attention_mask": shape (batch_size, num_modalities, seq_len)
                "modality": [
                    ("amino_acid", "nucleotide"),
                    ("SMILES", "nucleotide"),
                    ("amino_acid", "SMILES"),
                ]
            }
            Contains three independent samples with 2 representations each.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            The input batch containing input_ids, attention_mask, and modality information.
        index : int
            The index of the input to extract.

        Returns
        -------
        dict[str, Tensor | list[Modality]]
            A new batch containing only the specified input.
        """
        return {
            "input_ids": batch["input_ids"][:, index],
            "attention_mask": batch["attention_mask"][:, index],
            "modality": [
                modalities[index] if isinstance(modalities, (list, tuple)) else modalities
                for modalities in batch["modality"]
            ],
        }

    def _mlm_infonce_step(self, batch: dict[str, Tensor | list[Modality]], stage: Literal["train", "val"]) -> Tensor:
        """Combine MLM and InfoNCE losses for dual input training.

        This method combines masked language modeling and InfoNCE contrastive losses
        for training with two input modalities, weighted by contrastive_loss_weight.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            The batch containing two input modalities.
        stage : Literal["train", "val"]
            The current stage (training or validation).

        Returns
        -------
        Tensor
            The combined loss: (1 - contrastive_loss_weight) * MLM_loss + contrastive_loss_weight * InfoNCE_loss
        """
        batch_a = self._split_batch_by_index(batch, 0)
        batch_b = self._split_batch_by_index(batch, 1)

        # Compute losses based on weights
        mlm_loss = (
            self._compute_mlm_loss(batch_a, stage)
            if self.contrastive_loss_weight != 1.0
            else torch.tensor(0.0, device=self.device)
        )
        contrastive_loss = (
            self._compute_infonce_loss(batch_a, batch_b, stage)
            if self.contrastive_loss_weight > 0
            else torch.tensor(0.0, device=self.device)
        )

        return (1 - self.contrastive_loss_weight) * mlm_loss + self.contrastive_loss_weight * contrastive_loss

    def _mlm_symile_step(self, batch: dict[str, Tensor | list[Modality]], stage: Literal["train", "val"]) -> Tensor:
        """Combine MLM and Symile losses for multi-modal training.

        This method combines masked language modeling and Symile contrastive losses
        for training with more than two input modalities, weighted by contrastive_loss_weight.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            The batch containing multiple input modalities.
        stage : Literal["train", "val"]
            The current stage (training or validation).

        Returns
        -------
        Tensor
            The combined loss: (1 - contrastive_loss_weight) * MLM_loss + contrastive_loss_weight * Symile_loss
        """
        batches = [self._split_batch_by_index(batch, i) for i in range(batch["input_ids"].shape[1])]

        mlm_loss = (
            self._compute_mlm_loss(batches[0], stage)
            if self.contrastive_loss_weight != 1.0
            else torch.tensor(0.0, device=self.device)
        )
        symile_loss = (
            self._compute_symile_loss(batches, stage)
            if self.contrastive_loss_weight > 0
            else torch.tensor(0.0, device=self.device)
        )

        return (1 - self.contrastive_loss_weight) * mlm_loss + self.contrastive_loss_weight * symile_loss

    def _delegate_step_by_batch_shape(
        self, batch: dict[str, Tensor | list[Modality]], stage: Literal["train", "val"]
    ) -> Tensor:
        """Delegate the training/validation step based on batch shape.

        Parameters
        ----------
        batch : dict[str, Tensor | list[Modality]]
            The input batch containing input_ids, attention_mask, and modality information.
        stage : Literal["train", "val"]
            The current stage (training or validation).

        Returns
        -------
        Tensor
            The computed loss for the current step.

        Raises
        ------
        ValueError
            If the batch shape is not supported.
        """
        num_inputs = batch["input_ids"].shape[1]

        if num_inputs == 1:
            batch = self._split_batch_by_index(batch, 0)
            return self._compute_mlm_loss(batch, stage)
        elif num_inputs == 2:
            return self._mlm_infonce_step(batch, stage)
        elif num_inputs > 2:
            return self._mlm_symile_step(batch, stage)
        else:
            raise ValueError(f"Unsupported batch shape: {batch['input_ids'].shape}")

    def training_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        loss = self._delegate_step_by_batch_shape(batch, "train")
        self.log("train_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss

    def validation_step(self, batch: dict[str, Tensor | list[Modality]], batch_idx: int) -> Tensor:
        loss = self._delegate_step_by_batch_shape(batch, "val")
        self.log("val_loss", loss, rank_zero_only=True, sync_dist=True)

        return loss
