import random
from typing import Literal, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    MatthewsCorrCoef,
    MeanAbsoluteError,
    Precision,
    R2Score,
    Recall,
    SpearmanCorrCoef,
)

from ._utils import POOLERS, model_typer

# torch.set_float32_matmul_precision("medium")


class LobsterMLP(pl.LightningModule):
    def __init__(
        self,
        num_labels: int = 1,
        num_chains: int = 1,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        model_type: Literal["LobsterPMLM", "LobsterPCLM"] = "LobsterPMLM",
        ffd_dim: int = 64,
        pooler: Optional[Literal["mean", "attn", "cls"]] = "mean",
        lr: float = 1e-3,
        seed: int = 0,
        max_length: int = 512,
        reinit: bool = False,
        metric_average: str = "weighted",
        output_hidden: bool = False,
        freeze_encoder: bool = True,
        linear_probe: bool = False,
    ):
        """
        MLP head.
        Please make sure max_length is long enough to capture the sequence variation in your dataset!
        E.g. if all variation happens in ixs 561-588 (FLIP AAV), max_length should be at least 588 unless
        the dataset or a transform already truncates sequences to 512. (see AAV dataset)

        Parameters
        ----------
        num_labels: regression (1), binary classification (2), or multi class (2+)
        num_chains: number of chains in input data, used to adjust MLP size
        model_name: load a specific pre-trained model, e.g. 'esm2_t6_8M_UR50D'
        checkpoint: path to load a pretrained Lobster model
        ckpt_path: continue training LobsterMLP from this checkpoint
        model_type: pre-trained model class
        pooler: how embeddings are pooled (mean, attn, cls) [default: mean]
        reinit: use random embeddings rather than pre-trained for benchmarking
        freeze_encoder: freeze the base encoder weights
        linear_probe: use only a linear layer for probing

        """

        super().__init__()

        self._seed = seed
        # Set random seeds
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        model_cls = model_typer[model_type]
        self._num_labels = num_labels
        self._num_chains = num_chains
        self._max_length = max_length
        self._model_name = model_name
        self._lr = lr
        self._reinit = reinit
        self._metric_average = metric_average
        self._ffd_dim = ffd_dim
        self._checkpoint = checkpoint
        self._freeze_encoder = freeze_encoder
        self._linear_probe = linear_probe

        model = None
        if model_name is not None:
            model = model_cls(
                model_name=model_name, max_length=max_length
            )  # load a specific model, e.g. ESM2-8M
        if checkpoint is not None:
            if checkpoint.endswith(".pt"):
                assert (
                    model is not None
                ), "If checkpoint ends in .pt, please also specify model_name"
                model.model.load_state_dict(torch.load(checkpoint))
            else:
                model = model_cls.load_from_checkpoint(
                    checkpoint,
                    model_name=model_name,
                    max_length=max_length,
                )  # load specific pre-trained chkpt

        self.model_name = model_name
        self.model_type = model_type
        self.model = model
        self.pooler = POOLERS[pooler](config=self.model.model.config)
        self.pooler_name = pooler
        self.max_input_length = max_length
        self.output_hidden = output_hidden

        if self._freeze_encoder:
            for (
                _name,
                param,
            ) in self.model.named_parameters():  # freeze pre-trained encoder
                param.requires_grad = False

        # Randomly re-initialize the base encoder weights
        if self._reinit:
            print("Re-initializing base encoder weights")
            self.model.model.init_weights()

        hidden_size = self.model.config.hidden_size
        self._hidden_size = hidden_size

        # metrics
        if self._num_labels == 1:
            self.train_r2score = R2Score()
            self.val_r2score = R2Score()
            self.test_r2score = R2Score()
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()
            self.test_mae = MeanAbsoluteError()
            self.train_spearman = SpearmanCorrCoef()
            self.val_spearman = SpearmanCorrCoef()
            self.test_spearman = SpearmanCorrCoef()
        elif self._num_labels > 1:
            task = "binary" if self._num_labels == 2 else "multiclass"
            self.train_average_precision = AveragePrecision(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.val_average_precision = AveragePrecision(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.train_auroc = AUROC(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.val_auroc = AUROC(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.train_accuracy = Accuracy(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.val_accuracy = Accuracy(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.train_mcc = MatthewsCorrCoef(task=task, num_classes=self._num_labels)
            self.val_mcc = MatthewsCorrCoef(task=task, num_classes=self._num_labels)
            self.train_precision = Precision(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.val_precision = Precision(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.train_recall = Recall(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )
            self.val_recall = Recall(
                task=task, num_classes=self._num_labels, average=self._metric_average
            )

        self.loss = None
        match self._num_labels:
            case 1:
                self.loss = nn.MSELoss()
            case 2:
                self.loss = nn.BCEWithLogitsLoss()  # expects logits
            case _:
                self.loss = nn.CrossEntropyLoss()

        output_dim = self._num_labels if self._num_labels > 2 else 1
        if self._linear_probe:
            self.probe = nn.Linear(int(self._num_chains * hidden_size), output_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(int(self._num_chains * hidden_size), ffd_dim),
                nn.LayerNorm(ffd_dim),
                nn.ReLU(),
                nn.Linear(ffd_dim, ffd_dim),
                nn.LayerNorm(ffd_dim),
                nn.ReLU(),
                nn.Linear(ffd_dim, output_dim),
            )

        # Cache for base encoder embeddings
        self.embedding_cache = {}

        self.save_hyperparameters(logger=False)

    def forward(self, x):
        if self._linear_probe:
            output = self.probe(x)
        else:
            output = self.mlp(x)
        return output

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        if self._num_labels == 1:
            r2score = self.train_r2score(preds, targets)
            mae = self.train_mae(preds, targets)
            spearman = self.train_spearman(preds, targets)
            loss_dict = {
                "train/r2score": r2score,
                "train/mae": mae,
                "train/spearman": spearman,
            }
            self.log_dict(loss_dict)
        elif self._num_labels > 1:
            targets = targets.long()
            average_precision = self.train_average_precision(preds, targets)
            accuracy = self.train_accuracy(preds, targets)
            auroc = self.train_auroc(preds, targets)
            mcc = self.train_mcc(preds, targets)
            loss_dict = {
                "train/average_precision": average_precision,
                "train/accuracy": accuracy,
                "train/auroc": auroc,
                "train/mcc": mcc,
            }
            self.log_dict(loss_dict)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        if self._num_labels == 1:
            r2score = self.val_r2score(preds, targets)
            mae = self.val_mae(preds, targets)
            spearman = self.val_spearman(preds, targets)
            loss_dict = {
                "val/r2score": r2score,
                "val/mae": mae,
                "val/spearman": spearman,
            }
            self.log_dict(loss_dict)
        elif self._num_labels > 1:
            targets = targets.long()
            average_precision = self.val_average_precision(preds, targets)
            accuracy = self.val_accuracy(preds, targets)
            auroc = self.val_auroc(preds, targets)
            mcc = self.val_mcc(preds, targets)
            loss_dict = {
                "val/average_precision": average_precision,
                "val/accuracy": accuracy,
                "val/auroc": auroc,
                "val/mcc": mcc,
            }
            self.log_dict(loss_dict)

        self.log("val/loss", loss, prog_bar=True, on_step=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.output_hidden:
            _loss, preds, targets, hidden_states = self._compute_loss(batch)
            return preds, targets, hidden_states

        _loss, preds, targets = self._compute_loss(batch)
        # r2score = self.test_r2score(preds, targets)
        # mae = self.test_mae(preds, targets)
        # spearman = self.test_spearman(preds, targets)

        # loss_dict = {"test/r2score": r2score, "test/mae": mae, "test/spearman": spearman}
        # self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict)
        return preds, targets  # , loss_dict

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self._lr)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        # return [optimizer], [scheduler]
        return optimizer

    def _compute_loss(self, batch) -> Tuple[torch.Tensor]:
        sequences, ys = batch
        if self._num_labels < 3:  # regression or binary
            ys = ys[0].float()
            ys = torch.nan_to_num(
                ys, nan=0.0, posinf=0.0, neginf=0.0
            )  # fix missing vals
        else:
            ys = ys[0].long()
        all_hiddens = []
        input_mask = None  # Could optionally have input batch here

        for chains in sequences:  # could be heavy, light, single chain, etc.
            with torch.inference_mode():
                hidden_states = self.model.sequences_to_latents(chains)[-2].to(
                    ys
                )  # TODO - make desired layer an input

                if self.pooler is not None:
                    hidden_states = self.pooler(
                        hidden_states, input_mask=input_mask
                    )  # TODO -  add warning --> specify mask if seq length > self.max_input_length

                all_hiddens.append(hidden_states)
        all_hiddens = torch.concat(all_hiddens, dim=1).to(self.device).float()

        y_hat = self(all_hiddens).flatten()  # forward through MLP
        loss = self.loss(y_hat, ys)

        if self.output_hidden:
            return loss, y_hat, ys, all_hiddens

        return loss, y_hat, ys

    def _compute_loss_cached(self, batch):
        """
        Differs from method _compute_loss in two main ways: 1) Embeddings are cached for quicker embedding
        after epoch 0 and 2) assumes batch inputs are single chains
        TODO - we may want to combine this and compute_loss_cached with a flag for caching and a flag for single/multi-chain
        """
        sequences, ys = batch
        ys = ys[0].float()
        all_hiddens = []
        # Squeeze unnecessary dim in list
        if isinstance(sequences[0], tuple) or isinstance(sequences[0], list):
            sequences = sequences[0]

        for seq in sequences:
            # Check if embeddings are cached
            if seq in self.embedding_cache:
                hidden_states = self.embedding_cache[seq]
            else:
                with torch.inference_mode():
                    # Cast to list if single seq
                    if isinstance(seq, str):
                        seq = [seq]
                    if self.pooling == "mean":
                        # mean residue representation, 2nd to last layer features
                        hidden_states = (
                            self.model.sequences_to_latents(seq)[-2].mean(dim=1).to(ys)
                        )
                    elif self.pooling == "mean_nonzero":
                        hidden_states, attn_mask = self.model.sequences_to_latents(seq)
                        divisor = attn_mask.sum(axis=-1).unsqueeze(-1)
                        hidden_states = (hidden_states[-2].sum(axis=1) / divisor).to(ys)

                    self.embedding_cache[seq[0]] = hidden_states  # unpack seq from list
            all_hiddens.append(hidden_states)

        all_hiddens = torch.concat(all_hiddens, dim=0).to(self.device).float()

        y_hat = self(all_hiddens).flatten()  # forward through MLP
        y_hat = y_hat.float()

        # fix missing vals
        ys = torch.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)

        loss = F.mse_loss(y_hat, ys)

        return loss, y_hat, ys
