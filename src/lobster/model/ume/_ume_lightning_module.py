import logging
from collections.abc import Sequence
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

from ..losses import InfoNCELoss, SymileLoss
from ._ume import UME

logger = logging.getLogger(__name__)


class UMELightningModule(L.LightningModule):
    """Lightning wrapper for the Universal Molecular Encoder.

    This class provides a Lightning interface to the core UME PyTorch module,
    handling training, validation, loss computation, and optimization.

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
    weight_decay : float, default=0.0
        Weight decay for optimizer.

    Attributes
    ----------
    ume : UME
        The underlying UME PyTorch module.
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
        """Initialize the UME Lightning Module"""
        super().__init__()

        self.save_hyperparameters()

        # Initialize the core UME model
        self.ume = UME(
            model_name=model_name,
            max_length=max_length,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            mask_percentage=mask_percentage,
            scheduler=scheduler,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            model_kwargs=model_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            use_flash_attn=use_flash_attn,
            ckpt_path=ckpt_path,
            weight_decay=weight_decay,
        )

        # Store Lightning-specific parameters
        self.contrastive_loss_type = contrastive_loss_type
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_temperature = contrastive_temperature
        self.weight_decay = weight_decay
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
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

    def forward(self, *args, **kwargs):
        """Forward pass - delegate to UME model"""
        return self.ume(*args, **kwargs)

    def freeze(self) -> None:
        """Freeze the underlying UME model"""
        self.ume.freeze()

    def unfreeze(self) -> None:
        """Unfreeze the underlying UME model"""
        self.ume.unfreeze()

    @property
    def frozen(self) -> bool:
        """Check if the underlying UME model is frozen"""
        return self.ume.frozen

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension from UME model"""
        return self.ume.embedding_dim

    @property
    def modalities(self) -> list[str]:
        """Get supported modalities from UME model"""
        return self.ume.modalities

    def get_tokenizer(self, modality: ModalityType | Modality):
        """Get tokenizer from UME model"""
        return self.ume.get_tokenizer(modality)

    def get_vocab(self) -> dict[int, str]:
        """Get vocabulary from UME model"""
        return self.ume.get_vocab()

    def embed(self, inputs: dict[str, Tensor], aggregate: bool = True) -> Tensor:
        """Get embeddings from UME model"""
        return self.ume.embed(inputs, aggregate=aggregate)

    def embed_sequences(
        self, sequences: Sequence[str] | str, modality: ModalityType | Modality, aggregate: bool = True
    ) -> Tensor:
        """Embed sequences using UME model"""
        return self.ume.embed_sequences(sequences, modality, aggregate=aggregate)

    def compute_pseudo_likelihood(self, sequences: list[str], modality: Modality) -> list[float]:
        """Compute pseudo-likelihood using UME model"""
        return self.ume.compute_pseudo_likelihood(sequences, modality)

    def export_onnx(self, *args, **kwargs) -> None:
        """Export UME model to ONNX"""
        return self.ume.export_onnx(*args, **kwargs)

    def extract_batch_components(
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

    def split_combined_batch(
        self,
        combined_batch: dict[str, Tensor | list[Modality]],
    ) -> tuple[dict[str, Tensor | list[Modality]], ...]:
        """Split a combined batch of N inputs into N separate batches."""
        num_splits = combined_batch["input_ids"].shape[1]

        return tuple(self.extract_batch_components(combined_batch, i) for i in range(num_splits))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.ume.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )

        scheduler = transformers.get_scheduler(
            self.scheduler,
            optimizer,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.num_warmup_steps,
            scheduler_specific_kwargs=self.scheduler_kwargs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_infonce_loss(
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

    def process_batch_for_modality_metrics(
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

    def compute_weighted_loss(
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

    def compute_mlm_loss(
        self,
        batch: dict[str, Tensor | list[Modality]],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute masked language model loss."""
        # Prepare inputs for the model
        input_ids, attention_mask, cu_seqlens = self.ume.model._prepare_inputs(
            batch["input_ids"], batch["attention_mask"]
        )
        masked_input_ids, labels = self.ume.model._mask_inputs(input_ids)

        # Get model outputs
        hidden_states = self.ume.model.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=self.ume.max_length,
        )

        # Get logits from decoder and reshape for loss calculation
        logits = self.ume.model.decoder(hidden_states)
        logits = logits.view(-1, self.ume.model.config.vocab_size)  # (batch_size * sequence_length, vocab_size)
        labels = labels.view(-1)  # (batch_size * sequence_length)

        # Compute loss
        loss = self.ume.model.loss_fn(logits, labels)

        # Log overall metrics
        perplexity = torch.exp(loss)
        self.log(f"{stage}_perplexity", perplexity, rank_zero_only=True, sync_dist=True)

        # Process per-modality metrics
        modalities = batch["metadata"]["modality"] if "metadata" in batch else batch["modality"]
        self.process_batch_for_modality_metrics(logits, labels, modalities, stage)

        return loss

    def compute_symile_loss(
        self,
        embeddings: list[Tensor],
        stage: Literal["train", "val"],
    ) -> Tensor:
        """Compute Symile loss for a batch of N views of the same entity."""
        embeddings = [torch.nn.functional.normalize(embedding, dim=-1) for embedding in embeddings]

        return self.symile_loss_fn(embeddings)

    def contrastive_step(
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

        embeddings = [self.ume.embed(batch) for batch in batches]

        contrastive_loss_fn = (
            self.compute_symile_loss if self.contrastive_loss_type == "symile" else self.compute_infonce_loss
        )

        contrastive_loss = (
            contrastive_loss_fn(embeddings, stage=stage)
            if self.contrastive_loss_weight > 0
            else torch.tensor(0.0, device=self.device)
        )

        mlm_loss = (
            self.compute_mlm_loss(batches[0], stage)
            if self.contrastive_loss_weight != 1.0
            else torch.tensor(0.0, device=self.device)
        )

        return self.compute_weighted_loss(
            contrastive_loss=contrastive_loss,
            mlm_loss=mlm_loss,
            stage=stage,
        )

    def delegate_step_by_batch_shape(
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
            return self.compute_mlm_loss(batch, stage)

        batches = self.split_combined_batch(batch)

        return self.contrastive_step(*batches, stage=stage)

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
        loss = self.delegate_step_by_batch_shape(batch, "train")
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
        loss = self.delegate_step_by_batch_shape(batch, "val")
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
    ) -> "UMELightningModule":
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
        UMELightningModule
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
        model_name: Literal["ume-mini-base-12M", "ume-small-base-90M", "ume-medium-base-480M", "ume-large-base-740M"],
        *,
        device: str | None = None,
        use_flash_attn: bool | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> "UMELightningModule":
        """Load a pretrained UME model from a model name and wrap it in Lightning module.

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
            Additional keyword arguments to pass to the Lightning module.

        Returns
        -------
        UMELightningModule
            The loaded pretrained model wrapped in Lightning module.

        Examples
        --------
        >>> # Load UME-mini with default checkpoint
        >>> model = UMELightningModule.from_pretrained("ume-mini-base-12M")
        >>>
        >>> # Load UME-mini with specific device
        >>> model = UMELightningModule.from_pretrained("ume-mini-base-12M", device="cpu")
        >>>
        >>> # Load with custom cache directory
        >>> model = UMELightningModule.from_pretrained("ume-mini-base-12M", cache_dir="/path/to/cache")
        """
        # Load the core UME model
        ume_model = UME.from_pretrained(
            model_name=model_name,
            device=device,
            use_flash_attn=use_flash_attn,
            cache_dir=cache_dir,
            **kwargs,
        )

        # Create Lightning wrapper with the same hyperparameters
        lightning_module = cls(**ume_model.hparams, **kwargs)

        # Replace the UME instance with the loaded one
        lightning_module.ume = ume_model

        return lightning_module
