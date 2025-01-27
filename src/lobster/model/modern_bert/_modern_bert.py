import importlib.resources
from importlib.util import find_spec
import lightning.pytorch as pl
import torch
from torch import nn
from lobster.tokenization._pmlm_tokenizer import PmlmTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from lobster.tokenization._pmlm_tokenizer_transform import PmlmTokenizerTransform
from ._config import FlexBertConfig
from ._model import FlexBertModel, FlexBertPredictionHead

_FLASH_ATTN_AVAILABLE = False

if find_spec("flash_attn"):
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    _FLASH_ATTN_AVAILABLE = True

class FlexBERT(pl.LightningModule):

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1_000,
        tokenizer_dir: str = "pmlm_tokenizer",
        mask_percentage: float = 0.15,
        max_length: int = 512,
        **model_kwargs,
    ):
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._mask_percentage = mask_percentage
        self._max_length = max_length

        path = importlib.resources.files("lobster") / "assets" / tokenizer_dir
        self.tokenizer: PmlmTokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
        self.tokenize_transform = PmlmTokenizerTransform(
            path,
            padding="max_length",
            max_length=self._max_length,
            truncation=True,
        )

        config = FlexBertConfig(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id = self.tokenizer.pad_token_id,
            **model_kwargs,
        )
        self.model = FlexBertModel(config)

        self.decoder = nn.Sequential(
            FlexBertPredictionHead(config),
            nn.Linear(config.hidden_size, config.vocab_size)
        )

        assert _FLASH_ATTN_AVAILABLE, "flash_attn not available. This dependency is part of the flash extra"
        self.loss_fn = CrossEntropyLoss()
        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, sync_dist=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._num_training_steps,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def sequences_to_latents(self, sequences: list[str]) -> list[torch.Tensor]:
        transformed_sequences = self.tokenize_transform(sequences)
        input_ids = torch.concat([batch["input_ids"].squeeze(0) for batch in transformed_sequences]).to(self.device)
        attention_mask = torch.concat([batch["attention_mask"].squeeze(0) for batch in transformed_sequences]).to(self.device)
        seqlens = [batch["input_ids"].size(1) for batch in transformed_sequences]
        cu_seqlens = torch.tensor([sum(seqlens[:i]) for i in range(len(seqlens) + 1)], dtype=torch.int32).to(self.device)

        with torch.inference_mode():
            hidden_states = self.model(input_ids, attention_mask=attention_mask, cu_seqlens=cu_seqlens, max_seqlen=self._max_length)

        return [hidden_states[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(len(cu_seqlens) - 1)]

    def _compute_loss(self, batch):
        batch, _targets = batch

        tokens = batch["input_ids"].squeeze(1)
        B, length = tokens.shape
        tokens = tokens.view(-1)
        attention_mask=batch["attention_mask"].squeeze(1).view(-1)

        labels = tokens.clone()

        masked_tokens = self._mask_inputs(tokens)
        labels[masked_tokens != self.tokenizer.mask_token_id] = -100  # Ignore loss on unmasked tokens

        # Cumulative sequence lengths.
        # TODO: Probably we can/should throw away trailing <pad> tokens.
        cu_seqlens = torch.tensor([0] + [(i + 1) * length for i in range(B)], dtype=torch.int32).cuda()
        hidden_states = self.model(
            masked_tokens,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=self._max_length
        )

        logits = self.decoder(hidden_states)

        return self.loss_fn(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))

    def _mask_inputs(self, train_inputs: torch.Tensor):
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(train_inputs.shape, device=train_inputs.device)

        # create mask array
        mask_arr = (
            (rand < self._mask_percentage)
            * (train_inputs != self.tokenizer.cls_token_id)
            * (train_inputs != self.tokenizer.pad_token_id)
            * (train_inputs != self.tokenizer.eos_token_id)
        )  # don't mask cls, pad, eos

        masked_inputs = train_inputs.clone()
        masked_inputs[mask_arr] = self.tokenizer.mask_token_id

        return masked_inputs
