import torch
import torch.nn as nn
from torch import Tensor
from lightning import LightningModule

from .neobert_module import NeoBERTModule
from ._masking import mask_tokens


class NeoBERTLightningModule(LightningModule):
    def __init__(
        self,
        mask_token_id: int,
        cls_token_id: int,
        pad_token_id: int,
        eos_token_id: int,
        mask_probability: float = 0.2,
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        self.save_hyperparameters()

        super().__init__(**kwargs)

        self.mask_token_id = mask_token_id
        self.special_token_ids = [cls_token_id, pad_token_id, eos_token_id]
        self.mask_probability = mask_probability
        self.generator = generator

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.model = NeoBERTModule(pad_token_id=pad_token_id, **kwargs)

    def compute_mlm_loss(self, input_ids: Tensor, attention_mask: Tensor, **kwargs) -> Tensor:
        masked_inputs = mask_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_token_id=self.mask_token_id,
            mask_probability=self.mask_probability,
        )
        input_ids = masked_inputs["input_ids"]
        attention_mask = masked_inputs["attention_mask"]
        labels = masked_inputs["labels"]

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = logits["logits"]

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
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
