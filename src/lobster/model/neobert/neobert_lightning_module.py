import torch
import torch.nn as nn
from torch import Tensor
from lightning import LightningModule
import transformers
from .neobert_module import NeoBERTModule


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

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.model = NeoBERTModule(
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
            mask_probability=self.mask_probability,
            **model_kwargs,
        )

    def compute_mlm_loss(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        logits, labels = self.model.get_masked_logits_and_labels(input_ids, attention_mask, **kwargs)

        # Reshape logits and labels for cross-entropy loss
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
