from dataclasses import dataclass
import logging
from typing import Literal

import lightning as L
from torchmetrics import MeanAbsoluteError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
import torch
import torch.nn as nn

from ._heads import TaskConfig, FlexibleEncoderWithHeads
from lobster.post_train.unfreezing import set_unfrozen_layers


@dataclass
class PropertyRegressionConfig:
    """Configuration for training a property regression head on a generic encoder.

    Parameters
    - task_name: Name of the task/head; used to route outputs and metrics.
    - loss_function: Regression loss to use. Supported examples include
      'auto', 'l1', 'mse', 'huber', 'gaussian', 'mdn_gaussian'. For
      'gaussian', the head outputs two values per example (mean, log_scale).
      For 'mdn_gaussian', the head outputs parameters for a K-component
      Gaussian mixture; K is set by `mixture_components`.
    - hidden_sizes: Sizes of the MLP layers in the head. When None, a single
      linear layer is used.
    - dropout: Dropout probability applied inside the head MLP.
    - activation: Activation function for the head MLP. 'auto' picks a
      sensible default.
    - pooling: How to pool token embeddings into a sequence embedding.
      One of 'cls', 'mean', 'attn', 'weighted_mean'.
    - lr: Learning rate for the optimizer configured by this module.
    - weight_decay: Weight decay for the optimizer.
    - unfreeze_last_n_layers: Controls encoder layer unfreezing via
      `set_unfrozen_layers`:
        - None: leave `requires_grad` as-is
        - -1: unfreeze all encoder layers
        - 0: freeze all encoder layers
        - >0: unfreeze the last N encoder layers
    - mixture_components: Number of mixture components K for 'mdn_gaussian'.
    """

    task_name: str = "property"
    loss_function: str = "auto"
    hidden_sizes: list[int] | None = None
    dropout: float = 0.1
    activation: str = "auto"
    pooling: Literal["cls", "mean", "attn", "weighted_mean"] = "mean"
    lr: float = 1e-3
    weight_decay: float = 0.0
    unfreeze_last_n_layers: int | None = None
    mixture_components: int | None = None


class PropertyRegression(L.LightningModule):
    """LightningModule for training a regression head on top of any encoder.

    Args:
        encoder: The pretrained encoder used as the backbone.
        config: `PropertyRegressionConfig` controlling the head, loss,
            optimizer, pooling, and encoder unfreezing policy.
    """

    def __init__(self, encoder: nn.Module, *, config: PropertyRegressionConfig | None = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=[encoder])

        self.encoder = encoder
        cfg = config or PropertyRegressionConfig()
        self.cfg = cfg

        # Determine head output dimension based on loss
        head_output_dim = 1
        if cfg.loss_function == "gaussian":
            head_output_dim = 2  # mean, log_scale

        task = TaskConfig(
            name=cfg.task_name,
            output_dim=head_output_dim,
            task_type="regression",
            pooling=cfg.pooling,
            hidden_sizes=cfg.hidden_sizes,
            dropout=cfg.dropout,
            activation=cfg.activation,
            loss_function=cfg.loss_function,
            mixture_components=cfg.mixture_components,
        )

        # Resolve encoder hidden size for head construction
        hidden_size = None
        if hasattr(self.encoder, "embedding_dim"):
            hidden_size = getattr(self.encoder, "embedding_dim")
        elif hasattr(self.encoder, "config") and hasattr(self.encoder.config, "hidden_size"):
            hidden_size = self.encoder.config.hidden_size
        elif hasattr(self.encoder, "hidden_size"):
            hidden_size = getattr(self.encoder, "hidden_size")

        self.model = FlexibleEncoderWithHeads(
            encoder=self.encoder,
            task_configs=[task],
            hidden_size=hidden_size,
        )

        # Apply unfreezing if requested via config
        logging.getLogger(__name__).info(
            f"PropertyRegression: unfreeze_last_n_layers={cfg.unfreeze_last_n_layers}"
        )
        if cfg.unfreeze_last_n_layers is not None:
            n = int(cfg.unfreeze_last_n_layers)
            set_unfrozen_layers(self.encoder, n)

        self.loss_fns = self.model.get_loss_functions()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.train_spearman = SpearmanCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.train_pearson = PearsonCorrCoef()
        self.val_pearson = PearsonCorrCoef()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Return raw head output. For MDN this is params vector; for standard regression it's (B,1)
        return outputs[self.cfg.task_name]

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        preds = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])  # (B, P) or (B,1)
        targets = batch["targets"].to(preds.device)
        loss_fn = self.loss_fns[self.cfg.task_name]

        # Compute scalar prediction for metrics
        if self.cfg.loss_function == "mdn_gaussian":
            # Parse MDN params for D=1
            P = preds.shape[-1]
            if P % 3 != 0:
                raise ValueError(f"Expected MDN param size divisible by 3 for D=1, got {P}")
            K = P // 3
            logits = preds[:, :K]
            means = preds[:, K : 2 * K]
            # log_scales = preds[:, 2 * K : 3 * K]  # not needed for metrics
            weights = torch.softmax(logits, dim=-1)
            y_hat = torch.sum(weights * means, dim=-1)  # (B,)
            preds_for_loss = preds
        elif self.cfg.loss_function == "gaussian":
            # Natural Gaussian: preds = [mean, log_scale]
            if preds.shape[-1] != 2:
                raise ValueError(f"Gaussian loss expects head output dim=2, got {preds.shape[-1]}")
            y_hat = preds[..., 0]
            preds_for_loss = preds
        else:
            y_hat = preds.squeeze(-1)  # (B,)
            preds_for_loss = y_hat

        loss = loss_fn(preds_for_loss, targets)

        mae = self.train_mae if stage == "train" else self.val_mae
        r2 = self.train_r2 if stage == "train" else self.val_r2
        spearman = self.train_spearman if stage == "train" else self.val_spearman
        pearson = self.train_pearson if stage == "train" else self.val_pearson
        mae(y_hat, targets)
        r2(y_hat, targets)
        spearman(y_hat, targets)
        pearson(y_hat, targets)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_r2", r2, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_spearman", spearman, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_pearson", pearson, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], _: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return optimizer


