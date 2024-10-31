import random
from typing import Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torcheval.metrics import BinaryAUPRC
from torchmetrics import MatthewsCorrCoef, Precision, Recall

from ._utils import model_typer


class PPIClassifier(pl.LightningModule):
    def __init__(
        self,
        ffd_dim: int = 256,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,  # TODO - name this encoder_checkpoint for clarity?
        base_model_type: Literal["LobsterPMLM"] = "LobsterPMLM",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        seed: int = 0,
        freeze_encoder: bool = True,
        max_length: int = 512,
        # encoder_kwargs: Optional[dict] = {},
        ckpt_path: Optional[str] = None,  # dummy kwarg for compatibility with hydra (TODO - handle elsewhere?)
    ):
        """
        Classifier tasks for two-protein inputs (e.g. PPI).
        """
        super().__init__()
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._lr = lr
        self._seed = seed
        self._freeze_encoder = freeze_encoder
        self._max_length = max_length
        self._base_model_type = base_model_type
        self._model_cls = model_typer[base_model_type]

        # Set random seeds
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        # if model_name in PMLM_MODEL_NAMES:
        #     base_model = LobsterPMLM(model_name=model_name, **encoder_kwargs)
        #     if checkpoint is not None:
        #         base_model.load_from_checkpoint(checkpoint)

        # elif model_name in ESM_MODEL_NAMES:
        #     base_model = LobsterPMLM(model_name=model_name, **encoder_kwargs)

        base_model = None
        if model_name is not None:
            # TODO - This is redundant for .ckpt files, but necessary for .pt files. Maybe change logic here
            base_model = self._model_cls(
                model_name=model_name, max_length=max_length
            )  # load a specific model, e.g. ESM2-8M
        if (checkpoint is not None) and ("esm2" in model_name):
            if checkpoint.endswith(".pt"):
                assert base_model is not None, "If checkpoint ends in .pt, please also specify model_name"
                base_model.model.load_state_dict(torch.load(checkpoint))
            else:
                base_model = self._model_cls.load_from_checkpoint(
                    checkpoint,
                    model_name=model_name,
                    max_length=max_length,
                )  # load specific pre-trained chkpt

        self.model_name = model_name
        self.checkpoint = checkpoint
        self.base_model = base_model

        if self._freeze_encoder:
            for _name, param in self.base_model.named_parameters():
                param.requires_grad = False

        # NOTE - use hidden size from config by default
        self.hidden_size = self.base_model.model.config.hidden_size

        # metrics
        metric_params = {"task": "binary"}
        self.train_precision = Precision(**metric_params)
        self.val_precision = Precision(**metric_params)

        self.train_recall = Recall(**metric_params)
        self.val_recall = Recall(**metric_params)

        self.train_mc = MatthewsCorrCoef(**metric_params)
        self.val_mc = MatthewsCorrCoef(**metric_params)

        self.train_auprc = BinaryAUPRC()
        self.val_auprc = BinaryAUPRC()

        self.BCEWithLogitsLoss = BCEWithLogitsLoss()

        self.mlp = nn.Sequential(
            nn.Linear(int(self.hidden_size * 2), ffd_dim),  # seq1, seq2
            nn.LayerNorm(ffd_dim),
            nn.ReLU(),
            nn.Linear(ffd_dim, 1),
        )

        self.save_hyperparameters(logger=False)

    def forward(self, batch):
        tokens1, tokens2 = batch["tokens1"], batch["tokens2"]
        attns_a, attns_b = batch["attention_mask1"], batch["attention_mask2"]
        tokens_cat = torch.cat([tokens1, tokens2], dim=1)

        if "MLM" in self.model_name:
            preds = self.base_model.model(input_ids=tokens_cat, output_hidden_states=True)

        elif "esm2" in self.model_name:
            preds = self.base_model.model(tokens_cat, output_hidden_states=True)

        # Take mean embedding of last two layers, ignore padding
        divisor = attns_a.sum(axis=-1).unsqueeze(-1) + attns_b.sum(axis=-1).unsqueeze(-1)

        hidden_states_p = preds["hidden_states"][-2].sum(axis=1) / divisor
        hidden_states_u = preds["hidden_states"][-1].sum(axis=1) / divisor

        hidden_states = torch.cat([hidden_states_p, hidden_states_u], dim=1)

        y_hat = self.mlp(hidden_states).flatten()  # forward through MLP
        y_hat = y_hat.float()

        return y_hat

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        precision = self.train_precision(preds, targets)
        recall = self.train_recall(preds, targets)
        mc = self.train_mc(preds, targets)
        auprc = self.train_auprc.update(preds, targets).compute()

        loss_dict = {
            "train/precision": precision,
            "train/recall": recall,
            "train/mc": mc,
            "train/auprc": auprc,
        }
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        precision = self.val_precision(preds, targets)
        recall = self.val_recall(preds, targets)
        mc = self.val_mc(preds, targets)
        auprc = self.val_auprc.update(preds, targets).compute()

        loss_dict = {
            "val/precision": precision,
            "val/recall": recall,
            "val/mc": mc,
            "val/auprc": auprc,
        }
        self.log("val/loss", loss, prog_bar=True, on_step=True)
        self.log_dict(loss_dict)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        with torch.no_grad():
            preds = self.forward(batch)
            preds = torch.sigmoid(preds)
        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr, betas=(self._beta1, self._beta2), eps=self._eps)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        # return [optimizer], [scheduler]
        return optimizer

    def _compute_loss(self, batch):
        labels = batch["labels"]
        y_hat = self.forward(batch)

        # fix missing vals and scale targets
        ys = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        ys = ys.float()

        loss = self.BCEWithLogitsLoss(y_hat, ys)

        return loss, y_hat, ys


# Register this class into model_typer for later reference
model_typer["PPIClassifier"] = PPIClassifier
