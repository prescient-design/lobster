import logging
from typing import Literal
import lightning as L
import torch
import transformers
from torch import Tensor
from torchmetrics.text import Perplexity

from lobster.constants import (
    Modality,
)

from ..losses import InfoNCELoss, SymileLoss
from ._ume_module import UME

logger = logging.getLogger(__name__)


class UMELightningModule(L.LightningModule):
    """Lightning wrapper for the Universal Molecular Encoder.

    This class provides a Lightning interface to the core UME PyTorch module,
    handling training, validation, loss computation, and optimization.

    Parameters
    ----------
    model_name : Literal["mini", "small", "medium", "large"],
        default="mini"
        Name of the model to initialize.
    contrastive_loss_type : Literal[None, "symile", "clip", "disco_clip"], default=None
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
    **kwargs
        Additional keyword arguments to pass to the underlying UME model.
        See UME class documentation for available parameters:
        model_name, max_length, lr, beta1, beta2, eps, mask_percentage,
        scheduler, num_training_steps, num_warmup_steps, model_kwargs,
        scheduler_kwargs, use_flash_attn, ckpt_path, weight_decay.

    Attributes
    ----------
    ume :
        The underlying UME PyTorch module.
    """

    def __init__(
        self,
        contrastive_loss_type: Literal[None, "symile", "clip", "disco_clip"] = None,
        contrastive_loss_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
        **kwargs,
    ) -> None:
        """Initialize the UME Lightning Module"""
        super().__init__()

        self.save_hyperparameters()

        # Extract Lightning-specific parameters that aren't passed to UME
        self.contrastive_loss_type = contrastive_loss_type
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_temperature = contrastive_temperature

        # Extract parameters we need to store for Lightning optimizer/scheduler setup
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.lr = kwargs.get("lr", 1e-3)
        self.beta1 = kwargs.get("beta1", 0.9)
        self.beta2 = kwargs.get("beta2", 0.98)
        self.eps = kwargs.get("eps", 1e-12)
        self.num_training_steps = kwargs.get("num_training_steps", None)
        self.num_warmup_steps = kwargs.get("num_warmup_steps", 1_000)
        self.scheduler = kwargs.get("scheduler", "constant_with_warmup")
        self.scheduler_kwargs = kwargs.get("scheduler_kwargs", {}) or {}

        # Initialize the core UME model with all kwargs
        self.ume = UME(
            **kwargs,
        )

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
        **kwargs,
    ):
        """Load a model from a checkpoint with device-specific configuration.

        This method configures the model based on the specified or available device:
        - For CPU: Uses padded architecture with SDPA attention mask
        - For GPU: Uses unpadded architecture with Flash Attention

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        **kwargs
            Additional keyword arguments including use_flash_attn, device, and other
            parameters to pass to the parent class's load_from_checkpoint and UME model.

        Returns
        -------
        UMELightningModule
            The loaded model with appropriate device-specific configuration.

        Raises
        ------
        ValueError
            If an invalid device is specified.
        """
        # Extract device and use_flash_attn from kwargs
        device = kwargs.get("device")
        use_flash_attn = kwargs.get("use_flash_attn")

        # Determine device
        if device is not None:
            if device not in ["cpu", "cuda"]:
                raise ValueError(f"Invalid device: {device}. Must be one of ['cpu', 'cuda']")
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA device requested but not available")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs["device"] = device

        # Determine flash attention setting
        if use_flash_attn is None:
            use_flash_attn = device == "cuda"
            kwargs["use_flash_attn"] = use_flash_attn

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

        # Load the model using the parent class's method
        model = super().load_from_checkpoint(checkpoint_path, **kwargs)

        # Move model to specified device
        model = model.to(device)

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name: Literal["ume-mini-base-12M", "ume-small-base-90M", "ume-medium-base-480M", "ume-large-base-740M"],
        **kwargs,
    ):
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
        **kwargs
            Additional keyword arguments to pass to the Lightning module and underlying UME model.
            This includes parameters like device, use_flash_attn, cache_dir, and all other
            UME and Lightning-specific parameters.

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
            **kwargs,
        )

        # Map pretrained model name to Lightning model name
        model_name_mapping = {
            "ume-mini-base-12M": "mini",
            "ume-small-base-90M": "small",
            "ume-medium-base-480M": "medium",
            "ume-large-base-740M": "large",
        }
        lightning_model_name = model_name_mapping.get(model_name, "mini")

        # Create Lightning wrapper
        lightning_module = cls(model_name=lightning_model_name, **kwargs)

        # Replace the UME instance with the loaded one
        lightning_module.ume = ume_model

        return lightning_module
