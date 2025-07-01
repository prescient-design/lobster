from importlib.util import find_spec
from typing import Literal

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, get_scheduler

from ._config import FlexBertConfig
from ._model import FlexBertModel, FlexBertPredictionHead
from ._modern_bert_configuration import FLEXBERT_CONFIG_ARGS
from lobster.constants import SchedulerType

_FLASH_ATTN_AVAILABLE = False

if find_spec("flash_attn"):
    from flash_attn.losses.cross_entropy import CrossEntropyLoss

    _FLASH_ATTN_AVAILABLE = True
else:
    from torch.nn import CrossEntropyLoss


class FlexBERT(pl.LightningModule):
    def __init__(
        self,
        model_name: Literal["UME_mini", "UME_small", "UME_medium", "UME_large"] = "UME_mini",
        *,
        vocab_size: int | None = None,
        pad_token_id: int | None = None,
        mask_token_id: int | None = None,
        cls_token_id: int | None = None,
        eos_token_id: int | None = None,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int | None = None,
        num_warmup_steps: int = 1_000,
        mask_percentage: float = 0.25,
        max_length: int = 8192,
        scheduler: SchedulerType = "constant_with_warmup",
        model_kwargs: dict = None,
        scheduler_kwargs: dict = None,
        ckpt_path: str = None,
    ):
        """FlexBERT model for unsupervised pretraining.

        Parameters
        ----------
        model_name: str
            One of the keys in `FLEXBERT_CONFIG_ARGS`.
        vocab_size: int, optional
            The size of the vocabulary. Required if `tokenizer` is not provided.
        pad_token_id: int, optional
            The ID of the padding token. Required if `tokenizer` is not provided.
        mask_token_id: int, optional
            The ID of the mask token. Required if `tokenizer` is not provided.
        cls_token_id: int, optional
            The ID of the classification token. Required if `tokenizer` is not provided.
        eos_token_id: int, optional
            The ID of the end-of-sequence token. Required if `tokenizer` is not provided.
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, optional
            A pretrained tokenizer. Required if `vocab_size`, `pad_token_id`, `mask_token_id`, `cls_token_id`, and
            `eos_token_id` are not provided.
        lr: float, optional
            The learning rate.
        beta1: float, optional
            The beta1 parameter for the Adam optimizer.
        beta2: float, optional
            The beta2 parameter for the Adam optimizer.
        eps: float, optional
            The epsilon parameter for the Adam optimizer.
        num_training_steps: int, optional
            The total number of training steps.
        num_warmup_steps: int, optional
            The number of warmup steps.
        mask_percentage: float, optional
            The percentage of tokens to mask.
        max_length: int, optional
            The maximum sequence length.
        scheduler: str, optional
            The type of learning rate scheduler.
        model_kwargs: dict, optional
            Additional keyword arguments to pass to the model.
        scheduler_kwargs: dict, optional
            Additional keyword arguments to pass to the scheduler.
        ckpt_path: str, optional
            Unused, for compatiblity with lobster_train
        """

        super().__init__()
        self._model_name = model_name
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._mask_percentage = mask_percentage
        self.max_length = max_length
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        config_args = FLEXBERT_CONFIG_ARGS[model_name]
        model_kwargs = model_kwargs or {}

        # If flash-attn is not available but use_fa2 is True, warn and set to False
        if model_kwargs.get("use_fa2", True) and not _FLASH_ATTN_AVAILABLE:
            import warnings
            warnings.warn(
                "flash_attn not available but use_fa2=True. Setting use_fa2=False. "
                "This will use standard attention instead of flash-attn."
            )
            model_kwargs["use_fa2"] = False

        self.config = FlexBertConfig(
            **config_args,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            **model_kwargs,
        )
        self.model = FlexBertModel(self.config)

        self.decoder = nn.Sequential(
            FlexBertPredictionHead(self.config), nn.Linear(self.config.hidden_size, self.config.vocab_size)
        )

        self.loss_fn = CrossEntropyLoss()
        self.save_hyperparameters(logger=False)

        # Check that either a tokenizer OR vocab_size and special token IDs are provided
        if tokenizer is None and any(
            arg is None for arg in (vocab_size, pad_token_id, mask_token_id, cls_token_id, eos_token_id)
        ):
            raise ValueError("Either a tokenizer OR vocab_size and special token IDs must be provided")

        if tokenizer is not None:
            self.vocab_size = tokenizer.vocab_size
            self.pad_token_id = tokenizer.pad_token_id
            self.mask_token_id = tokenizer.mask_token_id
            self.cls_token_id = tokenizer.cls_token_id
            self.eos_token_id = tokenizer.eos_token_id
        else:
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
            self.mask_token_id = mask_token_id
            self.cls_token_id = cls_token_id
            self.eos_token_id = eos_token_id

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )

        scheduler = get_scheduler(
            self.scheduler, 
            optimizer, 
            num_training_steps=self._num_training_steps,
            num_warmup_steps=self._num_warmup_steps,
            scheduler_specific_kwargs=self.scheduler_kwargs
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def _prepare_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare inputs for the model by reshaping and calculating cumulative sequence lengths.
        Expects input_ids and attention_mask to be of shape (batch_size, 1, length).
        Returns reshaped input_ids and attention_mask of shape (batch_size * sequence_length,)
        and cumulative sequence lengths of shape (batch_size + 1,).
        """
        # Compute cumulative sequence lengths
        # input_ids and attention_mask are expected to be of shape (batch_size, 1, length)
        batch_size, length = input_ids.shape[0], input_ids.shape[2]
        cu_seqlens = torch.tensor([0] + [(i + 1) * length for i in range(batch_size)], dtype=torch.int32, device=self.device)

        # remove the middle dimension
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        # flatten to (batch_size * sequence_length)
        input_ids = input_ids.view(-1)  
        attention_mask = attention_mask.view(-1)

        assert (
            input_ids.max() < self.config.vocab_size
        ), f"Token ID {input_ids.max()} is out of vocabulary range {self.config.vocab_size}"
        
        return input_ids, attention_mask, cu_seqlens

    def _mask_inputs(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask inpust with random masking."""
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(input_ids.shape, device=input_ids.device)

        # create mask array
        # TODO: update this for special cls tokens that might be introduced with the new tokenizer
        mask_arr = (
            (rand < self._mask_percentage)
            * (input_ids != self.cls_token_id)
            * (input_ids != self.pad_token_id)
            * (input_ids != self.eos_token_id)
        )  # don't mask cls, pad, eos

        masked_inputs = input_ids.clone()
        masked_inputs[mask_arr] = self.mask_token_id

        labels  = input_ids.clone()
        labels[~mask_arr] = -100  # set unmasked tokens to -100 for loss calculation

        return masked_inputs, labels


    def _get_logits_and_labels(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Get logits and labels for training."""
        input_ids, attention_mask, cu_seqlens = self._prepare_inputs(
            batch["input_ids"], batch["attention_mask"]
        )
        
        # Mask input tokens
        masked_input_ids, labels = self._mask_inputs(input_ids)
        
        hidden_states = self.model(
            masked_input_ids, 
            attention_mask=attention_mask, 
            cu_seqlens=cu_seqlens, 
            max_seqlen=self.max_length
        )
        
        logits = self.decoder(hidden_states)
        
        logits = logits.view(-1, self.vocab_size) # (batch_size * sequence_length, vocab_size)
        labels = labels.view(-1) # (batch_size * sequence_length)
        
        return logits, labels
    
    def _compute_loss(self, batch: dict[str, Tensor] | tuple[dict[str, Tensor], Tensor]):
        """Compute the MLM loss for the given batch."""
        if isinstance(batch, tuple) and len(batch) == 2:
            batch, _ = batch

        logits, labels = self._get_logits_and_labels(batch)

        return self.loss_fn(logits, labels)


    def tokens_to_latents(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert input_ids to latents.
        
        Parameters
        ----------
        input_ids: torch.Tensor
            Input IDs of shape (batch_size, 1, length) for unpadded models or (batch_size, length) for padded models.
        attention_mask: torch.Tensor
            Attention mask of shape (batch_size, 1, length) for unpadded models or (batch_size, length) for padded models.
        
        Returns
        -------
        torch.Tensor
            Latents of shape (batch_size * sequence_length, hidden_size) for unpadded models
            or (batch_size, sequence_length, hidden_size) for padded models.
        """
        if self.config.padding == "unpadded":
            # For unpadded models, convert 3D to 2D if needed and let the encoder handle unpadding
            if input_ids.dim() == 3:
                input_ids = input_ids.squeeze(1)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            
            # Call the model with 2D tensors - the unpadded encoder will handle unpadding automatically
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        else:
            # For padded models, also ensure we have 2D tensors
            if input_ids.dim() == 3:
                input_ids = input_ids.squeeze(1)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)
            
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            