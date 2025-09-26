import logging
from typing import Literal

import torch
import torch.nn as nn
import transformers
from lightning import LightningModule
from torch import Tensor

from lobster.model.neobert import mask_tokens

from .ume_sequence_encoder import AuxiliaryTask, UMESequenceEncoderModule

logger = logging.getLogger(__name__)


class UMESequenceEncoderLightningModule(LightningModule):
    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int,
        special_token_ids: list[int],
        auxiliary_tasks: list[AuxiliaryTask] | None = None,
        mask_probability: float = 0.2,
        seed: int = 0,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
        scheduler: str = "constant",
        scheduler_kwargs: dict | None = None,
        encoder_kwargs: dict | None = None,
        use_shared_tokenizer: bool = False,
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
        self.use_shared_tokenizer = use_shared_tokenizer

        self.seed = seed
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.auxiliary_tasks = auxiliary_tasks
        self.auxiliary_task_loss_fns = {
            "regression": nn.MSELoss(),
        }

        self.encoder = UMESequenceEncoderModule(
            auxiliary_tasks=auxiliary_tasks,
            pad_token_id=self.pad_token_id,
            use_shared_tokenizer=self.use_shared_tokenizer,
            **encoder_kwargs or {},
        )

    def compute_mlm_loss(self, batch: dict[str, Tensor]) -> Tensor:
        masked_inputs = mask_tokens(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            mask_token_id=self.mask_token_id,
            mask_probability=self.mask_probability,
            special_token_ids=self.special_token_ids,
            seed=self.seed,
        )
        logits = self.encoder.neobert.get_logits(
            input_ids=masked_inputs["input_ids"], attention_mask=masked_inputs["attention_mask"]
        )

        # Reshape logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # Reshape labels: (batch_size, seq_len) -> (batch_size * seq_len,)
        logits = logits.view(-1, logits.size(-1))
        labels = masked_inputs["labels"].view(-1)

        return self.loss_fn(logits, labels)

    def compute_auxiliary_tasks_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.auxiliary_tasks is None:
            return {}

        if all(task.loss_weight == 0.0 for task in self.auxiliary_tasks):
            return {task.name: torch.tensor(0.0) for task in self.auxiliary_tasks}

        output = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], return_auxiliary_tasks=True
        )

        auxiliary_losses = {}

        for auxiliary_task in self.auxiliary_tasks:
            if auxiliary_task.name not in batch:
                raise ValueError(
                    f"Auxiliary task `{auxiliary_task.name}` labels not found in batch keys: {batch.keys()}"
                )

            labels = batch[auxiliary_task.name]
            logits = output[auxiliary_task.name]

            loss_fn = self.auxiliary_task_loss_fns[auxiliary_task.task_type]

            auxiliary_losses[auxiliary_task.name] = loss_fn(logits, labels)

        return auxiliary_losses

    def step(self, batch: dict[str, Tensor], batch_idx: int, stage: Literal["train", "val"]) -> Tensor:
        batch["input_ids"], batch["attention_mask"] = self.encoder.neobert.ensure_2d(
            batch["input_ids"], batch["attention_mask"]
        )
        batch_size = batch["input_ids"].shape[0]

        mlm_loss = self.compute_mlm_loss(batch)
        self.log(f"{stage}_mlm_loss", mlm_loss, sync_dist=True, rank_zero_only=True, batch_size=batch_size)

        auxiliary_losses = self.compute_auxiliary_tasks_loss(batch)
        total_loss = mlm_loss

        for auxiliary_task_name, auxiliary_loss in auxiliary_losses.items():
            auxiliary_task_info = next(task for task in self.auxiliary_tasks if task.name == auxiliary_task_name)
            loss_weight = auxiliary_task_info.loss_weight
            total_loss += loss_weight * auxiliary_loss

            self.log(
                f"{stage}_{auxiliary_task_name}_loss",
                auxiliary_loss,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=batch_size,
            )
            self.log(
                f"{stage}_{auxiliary_task_name}_weighted_loss",
                loss_weight * auxiliary_loss,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=batch_size,
            )

        self.log(f"{stage}_loss", total_loss, sync_dist=True, rank_zero_only=True, batch_size=batch_size)

        return total_loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self.step(batch, batch_idx, "train")
        self.log("train_loss", loss, sync_dist=True, rank_zero_only=True, batch_size=batch["input_ids"].shape[0])
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss = self.step(batch, batch_idx, "val")
        self.log("val_loss", loss, sync_dist=True, rank_zero_only=True, batch_size=batch["input_ids"].shape[0])
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
