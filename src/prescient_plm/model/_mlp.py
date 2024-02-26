import random
from typing import Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import MeanAbsoluteError, R2Score, SpearmanCorrCoef

from ._mlm import PrescientPMLM
from ._utils import model_typer

# torch.set_float32_matmul_precision("medium")


class RegressionHead(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 72,
        ffd_dim: int = 64,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        model_type: Literal["PrescientPMLM", "PrescientPRLM"] = "PrescientPMLM",
        pooling: str = "mean",
        lr: float = 1e-3,
        ckpt_path: str = None,
        seed: int = 0,
        max_length: int = 512,
        reinit: bool = False,
    ):
        """
        Regression head for PrescientPMLM.
        Please make sure max_length is long enough to capture the sequence variation in your dataset!
        E.g. if all variation happens in ixs 561-588 (FLIP AAV), max_length should be at least 588 unless
        the dataset or a transform already truncates sequences to 512. (see AAV dataset)

        Parameters
        ----------
        input_dim: hidden size of PMLM model
        pooling: how embeddings are pooled; currently only supports 'mean'

        """

        super().__init__()

        self._seed = seed
        # Set random seeds
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        model_cls = model_typer[model_type]

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
        self.input_dim = input_dim
        self.ffd_dim = ffd_dim
        self.checkpoint = checkpoint
        self.model = model
        self.pooling = pooling
        self._lr = lr
        self._ckpt_path = ckpt_path
        self._reinit = reinit

        for _name, param in self.model.named_parameters():  # freeze pre-trained encoder
            param.requires_grad = False

        # Randomly re-initialize the base encoder weights
        if self._reinit:
            print("Re-initializing base encoder weights")
            self.model.model.init_weights()

        # hidden_size = self.model.model.config.hidden_size

        # metrics
        self.train_r2score = R2Score()
        self.val_r2score = R2Score()
        self.test_r2score = R2Score()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.train_spearman = SpearmanCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.test_spearman = SpearmanCorrCoef()

        self.mlp = nn.Sequential(
            # nn.Linear(int(3 * hidden_size), ffd_dim),  # fv_heavy, fv_light, and antigen
            nn.Linear(input_dim, ffd_dim),
            nn.LayerNorm(ffd_dim),
            nn.ReLU(),
            nn.Linear(ffd_dim, 1),
        )

        # Cache for base encoder embeddings
        self.embedding_cache = {}

        self.save_hyperparameters(logger=False)

    def forward(self, x):
        output = self.mlp(x)
        return output

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        r2score = self.train_r2score(preds, targets)
        mae = self.train_mae(preds, targets)
        spearman = self.train_spearman(preds, targets)

        loss_dict = {"train/r2score": r2score, "train/mae": mae, "train/spearman": spearman}
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        r2score = self.val_r2score(preds, targets)
        mae = self.val_mae(preds, targets)
        spearman = self.val_spearman(preds, targets)

        loss_dict = {"val/r2score": r2score, "val/mae": mae, "val/spearman": spearman}
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        r2score = self.test_r2score(preds, targets)
        mae = self.test_mae(preds, targets)
        spearman = self.test_spearman(preds, targets)

        loss_dict = {"test/r2score": r2score, "test/mae": mae, "test/spearman": spearman}
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
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

    def _compute_loss(self, batch):
        sequences, ys = batch
        ys = ys[0].float()
        all_hiddens = []

        for chains in sequences:  # could be heavy, light, single chain, etc.
            with torch.inference_mode():
                # mean residue representation, 2nd to last layer features
                if self.pooling == "mean":
                    hidden_states = self.model.sequences_to_latents(chains)[-2].mean(dim=1).to(ys)
                # Mean token emb ignoring padding tokens
                elif self.pooling == "mean_nonzero":
                    hidden_states, attn_mask = self.model.sequences_to_latents(chains)
                    divisor = attn_mask.sum(axis=-1).unsqueeze(-1)
                    hidden_states = (hidden_states[-2].sum(axis=1) / divisor).to(ys)
                all_hiddens.append(hidden_states)
        all_hiddens = torch.concat(all_hiddens, dim=1).to(self.device).float()

        y_hat = self(all_hiddens).flatten()  # forward through MLP
        y_hat = y_hat.float()

        # fix missing vals
        ys = torch.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)

        loss = F.mse_loss(y_hat, ys)

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
                        hidden_states = self.model.sequences_to_latents(seq)[-2].mean(dim=1).to(ys)
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
