from importlib.util import find_spec
from typing import Literal

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, get_scheduler

from lobster.constants import Modality
from ._config import FlexBertConfig
from ._model import FlexBertModel, FlexBertPredictionHead
from ._modern_bert_configuration import FLEXBERT_CONFIG_ARGS

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

        assert _FLASH_ATTN_AVAILABLE, "flash_attn not available. This dependency is part of the flash extra"
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

    def tokens_to_latents(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Remove the middle dimension and flatten
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        batch_size, length = input_ids.shape  # Now shape is (batch_size, sequence_length)
        input_ids = input_ids.view(-1)  # Flatten to (batch_size * sequence_length)
        attention_mask = attention_mask.view(-1)

        # Create cumulative sequence lengths tensor
        cu_seqlens = torch.tensor(
            [0] + [(i + 1) * length for i in range(batch_size)], dtype=torch.int32, device=input_ids.device
        )

        with torch.inference_mode():
            hidden_states = self.model(
                input_ids, attention_mask=attention_mask, cu_seqlens=cu_seqlens, max_seqlen=self.max_length
            )

        return hidden_states

    def _compute_loss(self, batch, return_per_sample_loss: bool = False):
        if isinstance(batch, tuple) and len(batch) == 2:
            batch, _targets = batch

        tokens = batch["input_ids"].squeeze(1)
        B, length = tokens.shape
        tokens = tokens.view(-1)
        attention_mask = batch["attention_mask"].squeeze(1).view(-1)

        labels = tokens.clone()

        masked_tokens = self._mask_inputs(tokens)
        labels[masked_tokens != self.mask_token_id] = -100  # Ignore loss on unmasked tokens

        # Cumulative sequence lengths.
        cu_seqlens = torch.tensor([0] + [(i + 1) * length for i in range(B)], dtype=torch.int32).cuda()

        assert (
            masked_tokens.max() < self.config.vocab_size
        ), f"Token ID {masked_tokens.max()} is out of vocabulary range {self.config.vocab_size}"

        hidden_states = self.model(
            masked_tokens, attention_mask=attention_mask, cu_seqlens=cu_seqlens, max_seqlen=self.max_length
        )

        logits = self.decoder(hidden_states)

        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)

        batch_loss = self.loss_fn(logits, labels)

        if return_per_sample_loss:
            logits_reshaped = logits.view(B, length, self.vocab_size)
            labels_reshaped = labels.view(B, length)
            
            # Compute loss without reduction
            per_token_loss = torch.nn.functional.cross_entropy(
                logits_reshaped.transpose(1, 2), # (B, vocab_size, length)
                labels_reshaped,
                reduction="none",
            )
            
            # Mean loss per sample, ignoring extra tokens
            valid_tokens = (labels_reshaped != -100).float()
            token_count = valid_tokens.sum(dim=1).clamp(min=1.0)  # Avoid division by zero
            per_sample_loss = (per_token_loss * valid_tokens).sum(dim=1) / token_count

            return batch_loss, per_sample_loss

        return batch_loss

    def _mask_inputs(self, train_inputs: torch.Tensor):
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(train_inputs.shape, device=train_inputs.device)

        # create mask array
        # TODO: update this for special cls tokens that might be introduced with the new tokenizer
        mask_arr = (
            (rand < self._mask_percentage)
            * (train_inputs != self.cls_token_id)
            * (train_inputs != self.pad_token_id)
            * (train_inputs != self.eos_token_id)
        )  # don't mask cls, pad, eos

        masked_inputs = train_inputs.clone()
        masked_inputs[mask_arr] = self.mask_token_id

        return masked_inputs
