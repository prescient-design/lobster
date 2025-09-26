from dataclasses import dataclass
import logging
from typing import Literal

import torch
import torch.nn as nn
import transformers
from lightning import LightningModule
from torch import Tensor

from lobster.constants import Modality, to_modality
from lobster.model.neobert import mask_tokens

from .ume_interaction import UMEInteraction

logger = logging.getLogger(__name__)


@dataclass
class SpecialTokenIds:
    mask_token_id: int
    pad_token_id: int
    special_token_ids: list[int]


class UMEInteractionLightningModule(LightningModule):
    def __init__(
        self,
        special_token_mappings: dict[str | Modality, SpecialTokenIds],
        checkpoints: dict[str | Modality, str] | None = None,
        supported_modalities: list[str | Modality] | None = None,
        freeze_molecular_encoders: bool = False,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_ffn: int = 2048,
        dropout: float = 0.1,
        mask_probability: float = 0.2,
        seed: int = 0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        scheduler_kwargs: dict | None = None,
        encoder_kwargs: dict | None = None,
        ckpt_path: str | None = None,
        cache_dir: str | None = None,
    ):
        self.save_hyperparameters()

        super().__init__()

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

        self.special_token_mappings = {to_modality(key): value for key, value in special_token_mappings.items()}

        self.encoder = UMEInteraction(
            checkpoints=checkpoints,
            freeze_molecular_encoders=freeze_molecular_encoders,
            supported_modalities=supported_modalities,
            encoder_kwargs=encoder_kwargs,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_ffn=dim_ffn,
            dropout=dropout,
            cache_dir=cache_dir,
        )

    # def embed(self, inputs: dict[str, Tensor], aggregate: bool = True, ignore_padding: bool = True, **kwargs) -> Tensor:
    #     return self.encoder.embed(inputs=inputs, aggregate=aggregate, ignore_padding=ignore_padding, **kwargs)

    # def embed_sequences(
    #     self, sequences: Sequence[str] | str, modality: ModalityType | Modality = None, aggregate: bool = True
    # ) -> Tensor:
    #     return self.encoder.embed_sequences(sequences, modality=modality, aggregate=aggregate)

    def compute_mlm_loss(self, batch: dict[str, Tensor], stage: Literal["train", "val"]) -> Tensor:
        inputs: list[dict[str, Tensor | Modality]] = []
        labels: list[Tensor] = []

        for i in range(1, 3):
            input_ids = batch[f"input_ids{i}"]
            attention_mask = batch[f"attention_mask{i}"]

            # TODO: Support multiple modalities in a batch
            modalities: list[Modality | str] = batch[f"modality{i}"]

            if len(set(modalities)) > 1:
                raise NotImplementedError("Multiple modalities in a batch are not yet supported")

            modality = to_modality(modalities[0])

            special_token_ids = self.special_token_mappings[modality].special_token_ids
            mask_token_id = self.special_token_mappings[modality].mask_token_id

            input_ids, attention_mask = self.encoder.ensure_2d(input_ids, attention_mask)

            masked_inputs = mask_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mask_probability=self.mask_probability,
                special_token_ids=special_token_ids,
                mask_token_id=mask_token_id,
                seed=self.seed,
            )

            inputs.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "modality": modality,
                }
            )
            labels.append(masked_inputs["labels"])

        logits1, logits2 = self.encoder.get_logits(
            inputs1=inputs[0],
            inputs2=inputs[1],
        )

        # DEBUGGING: verify that the loss is higher without cross attention
        if stage == "val":
            no_cross_attention_logits1, no_cross_attention_logits2 = self.encoder.get_logits(
                inputs1=inputs[0],
                inputs2=inputs[1],
                use_cross_attention=False,
            )
            losses = []
            for logits, label in [
                (logits1, labels[0]),
                (logits2, labels[1]),
                (no_cross_attention_logits1, labels[0]),
                (no_cross_attention_logits2, labels[1]),
            ]:
                logits = logits.view(-1, logits.size(-1))
                label = label.view(-1)
                losses.append(self.loss_fn(logits, label))

            self.log(
                "val_mlm_loss_no_interaction",
                sum(losses) / len(losses),
                sync_dist=True,
                rank_zero_only=True,
                batch_size=batch["input_ids1"].shape[0],
            )

        losses = []

        for logits, label in [(logits1, labels[0]), (logits2, labels[1])]:
            logits = logits.view(-1, logits.size(-1))
            label = label.view(-1)
            losses.append(self.loss_fn(logits, label))

        return sum(losses) / len(losses)

    def step(self, batch: dict[str, Tensor], batch_idx: int, stage: Literal["train", "val"]) -> Tensor:
        batch_size = batch["input_ids1"].shape[0]

        mlm_loss = self.compute_mlm_loss(batch, stage)
        self.log(f"{stage}_mlm_loss", mlm_loss, sync_dist=True, rank_zero_only=True, batch_size=batch_size)

        return mlm_loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self.step(batch, batch_idx, "train")
        self.log("train_loss", loss, sync_dist=True, rank_zero_only=True, batch_size=batch["input_ids1"].shape[0])
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss, sync_dist=True, rank_zero_only=True, batch_size=batch["input_ids1"].shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
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
