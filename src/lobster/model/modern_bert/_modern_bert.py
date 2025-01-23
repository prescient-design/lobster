import importlib.resources
import lightning.pytorch as pl
import torch
from torch import nn
from lobster.tokenization._pmlm_tokenizer import PmlmTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from lobster.tokenization._pmlm_tokenizer_transform import PmlmTokenizerTransform
from ._config import FlexBertConfig
from ._model import FlexBertModel, FlexBertPredictionHead

from flash_attn.losses.cross_entropy import CrossEntropyLoss

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
        self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
        self.tokenize_transform = PmlmTokenizerTransform(
            tokenizer_dir=tokenizer_dir,
            padding="max_length",
            max_length=self._max_length,
            truncation=True,
        )

        config = FlexBertConfig(vocab_size=self.tokenizer.vocab_size, **model_kwargs)
        self.model = FlexBertModel(config)

        self.decoder = nn.Sequential(
            FlexBertPredictionHead(config),
            nn.Linear(config.hidden_size, config.vocab_size)
        )

        self.loss_fn = CrossEntropyLoss()

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

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        transformed_sequences = self.tokenize_transform(sequences)
        input_ids = torch.concat([batch["input_ids"].to(self.device) for batch in transformed_sequences])
        attention_mask = torch.concat([batch["attention_mask"].to(self.device) for batch in transformed_sequences])
        with torch.inference_mode():
            hidden_states = self.model(input_ids, attention_mask=attention_mask, max_seqlen=self._max_length)
        return hidden_states

    def _compute_loss(self, batch):
        tokens = batch["input_ids"]
        labels = tokens.clone()
        masked_tokens = self._mask_inputs(tokens)
        labels[masked_tokens != self.tokenizer.mask_token_id] = -100  # Ignore loss on unmasked tokens

        hidden_states = self.model(batch["input_ids"], attention_mask=batch["attention_mask"], max_seqlen=self._max_length)
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

        selection = []  # masked token positions

        for i in range(train_inputs.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

        masked_inputs = train_inputs.clone()
        for i in range(train_inputs.shape[0]):
            masked_inputs[i, selection[i]] = self.tokenizer.mask_token_id

        return masked_inputs
