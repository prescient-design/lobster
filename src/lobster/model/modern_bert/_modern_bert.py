from importlib.util import find_spec
from typing import Literal, Sequence

import lightning.pytorch as pl
import torch
from torch import nn
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import lightning.fabric.utilities.throughput
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf 

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
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1_000,
        mask_percentage: float = 0.25,
        max_length: int = 512,
        scheduler_cfg: DictConfig = None,
        **model_kwargs,
    ):
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
        self.scheduler_cfg = scheduler_cfg

        config_args = FLEXBERT_CONFIG_ARGS[model_name]

        self.config = FlexBertConfig(
            **config_args,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            **model_kwargs,
        )
        self.model = FlexBertModel(self.config)

        self.decoder = nn.Sequential(
            FlexBertPredictionHead(self.config),
            nn.Linear(self.config.hidden_size, self.config.vocab_size)
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

        # Initialize the scheduler using Hydra
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)

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
        cu_seqlens = torch.tensor([0] + [(i + 1) * length for i in range(batch_size)], 
                                dtype=torch.int32,
                                device=input_ids.device)
        
        with torch.inference_mode():
            hidden_states = self.model(
                input_ids,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=self.max_length
            )
        
        return hidden_states

    def _compute_loss(self, batch):

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
        # TODO: Probably we can/should throw away trailing <pad> tokens.
        cu_seqlens = torch.tensor([0] + [(i + 1) * length for i in range(B)], dtype=torch.int32).cuda()

        assert masked_tokens.max() < self.config.vocab_size, f"Token ID {masked_tokens.max()} is out of vocabulary range {self.config.vocab_size}"

        hidden_states = self.model(
            masked_tokens,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=self.max_length
        )

        logits = self.decoder(hidden_states)

        return self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

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
