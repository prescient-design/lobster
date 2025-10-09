from dataclasses import dataclass
import logging
from typing import Literal

import lightning as L
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import torch
import torch.nn as nn

from ._heads import TaskConfig, FlexibleEncoderWithHeads
from lobster.post_train.unfreezing import set_unfrozen_layers


@dataclass
class PropertyClassificationConfig:
    """Configuration for training a property classification head on a generic encoder.

    Parameters
    ----------
    task_name : str
        Name of the task/head; used to route outputs and metrics.
    num_classes : int
        Number of classes. For binary classification, use 2.
    loss_function : str
        Classification loss to use. Supported examples include
        'auto', 'bce', 'cross_entropy', 'focal'.
    hidden_sizes : list[int] | None
        Sizes of the MLP layers in the head. When None, a single
        linear layer is used.
    dropout : float
        Dropout probability applied inside the head MLP.
    activation : str
        Activation function for the head MLP. 'auto' picks a
        sensible default.
    pooling : Literal["cls", "mean", "attn", "weighted_mean"]
        How to pool token embeddings into a sequence embedding.
    lr : float
        Learning rate for the optimizer configured by this module.
    weight_decay : float
        Weight decay for the optimizer.
    unfreeze_last_n_layers : int | None
        Controls encoder layer unfreezing via `set_unfrozen_layers`:
        - None: leave `requires_grad` as-is
        - -1: unfreeze all encoder layers
        - 0: freeze all encoder layers
        - >0: unfreeze the last N encoder layers
    """

    task_name: str = "property"
    num_classes: int = 2
    loss_function: str = "auto"
    hidden_sizes: list[int] | None = None
    dropout: float = 0.1
    activation: str = "auto"
    pooling: Literal["cls", "mean", "attn", "weighted_mean"] = "mean"
    lr: float = 1e-3
    weight_decay: float = 0.0
    unfreeze_last_n_layers: int | None = None


class PropertyClassification(L.LightningModule):
    """LightningModule for training a classification head on top of any encoder.

    Args
    ----
    encoder : nn.Module
        The pretrained encoder used as the backbone.
    config : PropertyClassificationConfig
        Configuration controlling the head, loss, optimizer, pooling,
        and encoder unfreezing policy.
    """

    def __init__(self, encoder: nn.Module, *, config: PropertyClassificationConfig | None = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        cfg = config or PropertyClassificationConfig()
        self.cfg = cfg

        # Determine task type and output dimension
        if cfg.num_classes == 2:
            task_type = "binary_classification"
            head_output_dim = 1
        else:
            task_type = "multiclass_classification"
            head_output_dim = cfg.num_classes

        task = TaskConfig(
            name=cfg.task_name,
            output_dim=head_output_dim,
            task_type=task_type,
            pooling=cfg.pooling,
            hidden_sizes=cfg.hidden_sizes,
            dropout=cfg.dropout,
            activation=cfg.activation,
            loss_function=cfg.loss_function,
        )

        # Resolve encoder hidden size for head construction
        hidden_size = None
        if hasattr(self.encoder, "embedding_dim"):
            hidden_size = self.encoder.embedding_dim
        elif hasattr(self.encoder, "config") and hasattr(self.encoder.config, "hidden_size"):
            hidden_size = self.encoder.config.hidden_size
        elif hasattr(self.encoder, "hidden_size"):
            hidden_size = self.encoder.hidden_size

        self.model = FlexibleEncoderWithHeads(
            encoder=self.encoder,
            task_configs=[task],
            hidden_size=hidden_size,
        )

        # Apply unfreezing if requested via config
        logging.getLogger(__name__).info(f"PropertyClassification: unfreeze_last_n_layers={cfg.unfreeze_last_n_layers}")
        if cfg.unfreeze_last_n_layers is not None:
            n = int(cfg.unfreeze_last_n_layers)
            set_unfrozen_layers(self.encoder, n)

        self.loss_fns = self.model.get_loss_functions()

        # Metrics for binary classification
        if cfg.num_classes == 2:
            task_metric = "binary"
        else:
            task_metric = "multiclass"

        self.train_acc = Accuracy(
            task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None
        )
        self.val_acc = Accuracy(task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None)
        self.train_precision = Precision(
            task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None
        )
        self.val_precision = Precision(
            task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None
        )
        self.train_recall = Recall(
            task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None
        )
        self.val_recall = Recall(task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None)
        self.train_f1 = F1Score(task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None)
        self.val_f1 = F1Score(task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None)
        self.train_auroc = AUROC(task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None)
        self.val_auroc = AUROC(task=task_metric, num_classes=cfg.num_classes if task_metric == "multiclass" else None)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[self.cfg.task_name]

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        targets = batch["targets"].to(logits.device)
        loss_fn = self.loss_fns[self.cfg.task_name]

        # For binary classification, logits are (B,) and need to be passed through sigmoid for metrics
        if self.cfg.num_classes == 2:
            loss = loss_fn(logits, targets.float())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
        else:
            # For multiclass, logits are (B, C)
            loss = loss_fn(logits, targets.long())
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        # Update metrics
        acc = self.train_acc if stage == "train" else self.val_acc
        precision = self.train_precision if stage == "train" else self.val_precision
        recall = self.train_recall if stage == "train" else self.val_recall
        f1 = self.train_f1 if stage == "train" else self.val_f1
        auroc = self.train_auroc if stage == "train" else self.val_auroc

        acc(preds, targets)
        precision(preds, targets)
        recall(preds, targets)
        f1(preds, targets)
        auroc(probs if self.cfg.num_classes == 2 else probs, targets)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_precision", precision, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_recall", recall, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_f1", f1, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_auroc", auroc, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return optimizer
