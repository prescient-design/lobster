import torch
import torch.nn as nn
from torch import Tensor
from collections.abc import Sequence
from lightning import LightningModule
import transformers

from lobster.tokenization import UMETokenizerTransform
from lobster.constants import ModalityType, Modality

from .neobert_module import NeoBERTModule
from ._masking import mask_tokens


class NeoBERTLightningModule(LightningModule):
    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        special_token_ids: list[int],
        mask_probability: float = 0.2,
        seed: int = 0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        scheduler: str = "constant",
        scheduler_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
        ckpt_path: str | None = None,
    ):
        self.save_hyperparameters()

        super().__init__()

        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_token_ids = special_token_ids
        self.mask_probability = mask_probability
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}

        self.seed = seed
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.model = NeoBERTModule(
            pad_token_id=self.pad_token_id,
            **model_kwargs,
        )

    def embed(self, inputs: dict[str, Tensor], aggregate: bool = True, ignore_padding: bool = True, **kwargs) -> Tensor:
        if not all(k in inputs for k in {"input_ids", "attention_mask"}):
            raise ValueError("Missing required keys in inputs: 'input_ids' or 'attention_mask'")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if not aggregate:
            return output["last_hidden_state"]

        if not ignore_padding:
            return output["last_hidden_state"].mean(dim=1)

        mask = attention_mask.to(dtype=output["last_hidden_state"].dtype).unsqueeze(-1)

        masked_embeddings = output["last_hidden_state"] * mask

        sum_embeddings = masked_embeddings.sum(dim=1)
        token_counts = mask.sum(dim=1)

        return sum_embeddings / token_counts

    def embed_sequences(
        self, sequences: Sequence[str] | str, modality: ModalityType | Modality, aggregate: bool = True
    ) -> Tensor:
        if isinstance(sequences, str):
            sequences = [sequences]

        tokenizer_transform = UMETokenizerTransform(modality=modality, max_length=self.model.config.max_length)
        encoded_batch = tokenizer_transform(sequences)

        encoded_batch = {
            "input_ids": encoded_batch["input_ids"].to(self.device),
            "attention_mask": encoded_batch["attention_mask"].to(self.device),
        }

        return self.embed(encoded_batch, aggregate=aggregate)

    def compute_mlm_loss(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        if (
            input_ids.dim() == 3
            and input_ids.shape[1] == 1
            and attention_mask.dim() == 3
            and attention_mask.shape[1] == 1
        ):
            input_ids = input_ids.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

        assert input_ids.dim() == 2, "Input IDs must have shape: (batch_size, seq_len)"
        assert attention_mask.dim() == 2, "Attention mask must have shape: (batch_size, seq_len)"

        masked_inputs = mask_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            mask_probability=self.mask_probability,
            special_token_ids=self.special_token_ids,
            seed=self.seed,
        )
        input_ids = masked_inputs["input_ids"]
        attention_mask = masked_inputs["attention_mask"]
        labels = masked_inputs["labels"]

        logits = self.model.get_logits(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # labels: (batch_size, seq_len) -> (batch_size * seq_len,)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        return self.loss_fn(logits, labels)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self.compute_mlm_loss(batch["input_ids"], batch["attention_mask"])
        self.log("train_loss", loss, sync_dist=True, rank_zero_only=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self.compute_mlm_loss(batch["input_ids"], batch["attention_mask"])
        self.log("val_loss", loss, sync_dist=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        scheduler = transformers.get_scheduler(
            self.scheduler,
            optimizer,
            num_training_steps=self.scheduler_kwargs.pop("num_training_steps", None),
            num_warmup_steps=self.scheduler_kwargs.pop("num_warmup_steps", None),
            scheduler_specific_kwargs=self.scheduler_kwargs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
