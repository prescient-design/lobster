import random
from typing import Literal, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchmetrics import (
    MeanAbsoluteError,
    R2Score,
    SpearmanCorrCoef,
)
from torchvision.transforms import Resize

from ._utils import model_typer


class DyAbModel(pl.LightningModule):
    def __init__(
        self,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        model_type: Literal["LobsterPMLM", "LobsterPCLM"] = "LobsterPMLM",
        lr: float = 1e-3,
        seed: int = 0,
        max_length: int = 512,
        reinit: bool = False,
        metric_average: str = "weighted",
        output_hidden: bool = False,
        freeze_encoder: bool = True,
        embedding_img_size: int = 192,
        diff_channel_0: Literal["diff", "add", "mul", "div"] = "diff",
        diff_channel_1: Optional[Literal["sub", "add", "mul", "div"]] = None,
        diff_channel_2: Optional[Literal["diff", "add", "mul", "div"]] = None,
    ):
        """
        DyAb head.
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
        reinit: use random embeddings rather than pre-trained for benchmarking
        freeze_encoder: freeze the base encoder weights
        linear_probe: use only a linear layer for probing
        embedding_img_size: size of the image to be fed into the resnet
        diff_channel_{n}: how to create a difference of embeddings for channel {n}
            of the image to feed to the resnet.

        """
        super().__init__()

        self._seed = seed
        # Set random seeds
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        model_cls = model_typer[model_type]
        self._max_length = max_length
        self._model_name = model_name
        self._lr = lr
        self._reinit = reinit
        self._metric_average = metric_average
        self._checkpoint = checkpoint
        self._freeze_encoder = freeze_encoder
        self._embedding_img_size = embedding_img_size

        self._resize = Resize((self._embedding_img_size, self._embedding_img_size))
        self._diff_channel_0 = diff_channel_0
        self._diff_channel_1 = diff_channel_1
        self._diff_channel_2 = diff_channel_2

        if model_name is None and checkpoint is None:
            model_name = "esm2_t6_8M_UR50D"

        model = None
        if model_name is not None:
            model = model_cls(model_name=model_name, max_length=max_length)  # load a specific model, e.g. ESM2-8M
        if checkpoint is not None:
            if checkpoint.endswith(".pt"):
                assert model is not None, "If checkpoint ends in .pt, please also specify model_name"
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
        self.max_input_length = max_length
        self.output_hidden = output_hidden
        # Cache for base encoder embeddings
        self.embedding_cache = {}

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

        self._max_encoder_length = model.config.max_length
        self._hidden_size = self.model.config.hidden_size

        self.train_r2score = R2Score()
        self.val_r2score = R2Score()
        self.test_r2score = R2Score()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()
        self.train_spearman = SpearmanCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.test_spearman = SpearmanCorrCoef()

        ### Resnet
        self.resnet = models.resnet18(pretrained=True)
        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, 1)  # regression
        self.loss = nn.MSELoss(reduction="sum")

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        r2score = self.train_r2score(preds, targets)
        mae = self.train_mae(preds, targets)
        spearman = self.train_spearman(preds, targets)
        loss_dict = {
            "train/r2score": r2score,
            "train/mae": mae,
            "train/spearman": spearman,
        }
        self.log_dict(loss_dict)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        r2score = self.val_r2score(preds, targets)
        mae = self.val_mae(preds, targets)
        spearman = self.val_spearman(preds, targets)
        loss_dict = {
            "val/r2score": r2score,
            "val/mae": mae,
            "val/spearman": spearman,
        }
        self.log_dict(loss_dict)

        self.log("val/loss", loss, prog_bar=True, on_step=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch)
        r2score = self.test_r2score(preds, targets)
        mae = self.test_mae(preds, targets)
        spearman = self.test_spearman(preds, targets)
        loss_dict = {
            "test/r2score": r2score,
            "test/mae": mae,
            "test/spearman": spearman,
        }
        self.log_dict(loss_dict)

        self.log("test/loss", loss, prog_bar=True, on_step=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.output_hidden:
            _, preds, targets, hidden_states = self._compute_loss(batch)
            return preds, targets, hidden_states

        _, preds, targets = self._compute_loss(batch)
        return preds, targets  # , loss_dict

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self._lr)
        return optimizer

    def _compute_loss(self, batch, layer: int = -2):
        sequences1, sequences2, y1, y2 = batch
        # y1 = y1[0].float()
        y1 = torch.nan_to_num(y1, nan=0.0, posinf=0.0, neginf=0.0)  # fix missing vals
        y2 = torch.nan_to_num(y2, nan=0.0, posinf=0.0, neginf=0.0)  # fix missing vals

        if len(sequences1) == 2:  # concat multiple chains
            sequences1 = [s1 + "." + s2 for s1, s2 in zip(*sequences1)]
        if len(sequences2) == 2:  # concat multiple chains
            sequences2 = [s1 + "." + s2 for s1, s2 in zip(*sequences2)]

        for seq1, seq2 in zip(sequences1, sequences2):
            if seq1 not in self.embedding_cache:
                with torch.inference_mode():
                    hidden_states = self.model.sequences_to_latents([seq1])[layer].to(self.device).float()
                self.embedding_cache[seq1] = hidden_states
            if seq2 not in self.embedding_cache:
                with torch.inference_mode():
                    hidden_states = self.model.sequences_to_latents([seq2])[layer].to(self.device).float()
                self.embedding_cache[seq2] = hidden_states

        embeddings1 = torch.concat([self.embedding_cache[seq] for seq in sequences1], dim=0)
        embeddings2 = torch.concat([self.embedding_cache[seq] for seq in sequences2], dim=0)
        ys = y1 - y2  # subtract labels
        ys = ys.float()

        embedding_image = self._resize_embeddings(embeddings1, embeddings2)
        preds = self.resnet(embedding_image).squeeze().float()

        loss = self.loss(preds, ys)

        return loss, preds, ys  # embeddings1, embeddings2, embedding_image

    def _resize_embeddings(self, embeddings1, embeddings2):
        """
        Resize embeddings for Resnet.

        Arguments:
        ---------
        embeddings1: torch.Tensor  [B, L, H]
            Embeddings to resize
        embeddings2: torch.Tensor  [B, L, H]
            Embeddings to resize

        """
        img_dim = int(self._embedding_img_size)
        B = embeddings1.shape[0]
        input_image = torch.zeros((B, 3, img_dim, img_dim), device=self.device)

        for channel, op in enumerate([self._diff_channel_0, self._diff_channel_1, self._diff_channel_2]):
            if op is not None:
                if op == "diff":
                    embeddings = embeddings1 - embeddings2
                elif op == "add":
                    embeddings = embeddings1 + embeddings2
                elif op == "mul":
                    embeddings = embeddings1 * embeddings2
                elif op == "div":
                    embeddings = embeddings1 / embeddings2
                else:
                    raise ValueError(f"Invalid operation {op}")
            else:
                embeddings = torch.zeros(embeddings1.shape)

            embeddings = self._resize(embeddings)

            embeddings -= torch.amin(embeddings)

            # check that embeddings are not all zeros
            if torch.amax(embeddings) > 0:
                embeddings /= torch.amax(embeddings)

            embeddings = self._resize(embeddings)

            input_image[:, channel, :, :] += embeddings

        return input_image
