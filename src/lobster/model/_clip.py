from os import PathLike
from typing import Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import EsmForMaskedLM

from lobster.model._mlm import LobsterPMLM


class MiniCLIP(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path: Union[str, PathLike], lr: float = 1e-3):
        super().__init__()
        self._lr = lr

        # pre-trained encoder
        if "esm2" in pretrained_model_name_or_path:
            self.model = EsmForMaskedLM.from_pretrained(f"facebook/{pretrained_model_name_or_path}")
            input_dim = self.model.config.hidden_size
        else:
            self.model = LobsterPMLM.load_from_checkpoint(pretrained_model_name_or_path)
            input_dim = self.model.model.config.hidden_size
        self._input_dim = input_dim

        for _name, param in self.model.named_parameters():
            param.requires_grad = False

        # protein (antigen) encoder
        self.prot_embedder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
        )

        # binder (antibody) encoder
        self.binder_embedder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
        )

        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        batch_size = batch[0][0].shape[0]
        loss, _logits = self._compute_loss(batch)

        self.log("train_loss", loss, sync_dist=True, batch_size=batch_size)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        batch_size = batch[0][0].shape[0]
        loss, _logits = self._compute_loss(batch)

        self.log("val_loss", loss, sync_dist=True, batch_size=batch_size)

        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx):
        batch_size = batch[0][0].shape[0]
        _loss, logits = self._compute_loss(batch)
        labels = torch.arange(batch_size).to(self.device)

        # prediction of binder for each partner
        binder_predictions = logits.argmax(dim=0)
        # prediction of partners for each binder
        partner_predictions = logits.argmax(dim=1)

        binder_ranks = logits.argsort(dim=0).diag() + 1
        binder_mrr = (binder_ranks).float().pow(-1).mean()  # Mean Reciprocal Rank

        partner_ranks = logits.argsort(dim=1).diag() + 1
        partner_mrr = (partner_ranks).float().pow(-1).mean()

        partner_accuracy = partner_predictions.eq(labels).float().mean()
        binder_accuracy = binder_predictions.eq(labels).float().mean()

        k = int(logits.shape[0] / 10)
        binder_topk_accuracy = (
            torch.any((logits.topk(k, dim=0).indices - labels.reshape(1, -1)) == 0, dim=0).sum() / logits.shape[0]
        )
        partner_topk_accuracy = (
            torch.any((logits.topk(k, dim=1).indices - labels.reshape(-1, 1)) == 0, dim=1).sum() / logits.shape[0]
        )

        metrics_dict = {
            "binder_mrr": binder_mrr,
            "partner_mrr": partner_mrr,
            "partner_accuracy": partner_accuracy,
            "binder_accuracy": binder_accuracy,
            "binder_topk_accuracy": binder_topk_accuracy,
            "partner_topk_accuracy": partner_topk_accuracy,
        }

        return metrics_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    def _compute_loss(self, batch):
        if len(batch[0]) == 3:  # antibody / antigen
            fv_heavy, fv_light, antigen = batch[0]
            fv_heavy_embedding = self.model(input_ids=fv_heavy.squeeze(), output_hidden_states=True)["hidden_states"][
                -1
            ]
            fv_light_embedding = self.model(input_ids=fv_light.squeeze(), output_hidden_states=True)["hidden_states"][
                -1
            ]
            binder_embedding = F.normalize(
                self.binder_embedder(fv_heavy_embedding.mean(dim=1) + fv_light_embedding.mean(dim=1))
            )

        elif len(batch[0]) == 2:  # peptide / antigen
            binder, antigen = batch[0]
            binder_esm_embedding = self.model(input_ids=binder.squeeze(), output_hidden_states=True)["hidden_states"][
                -1
            ]
            binder_embedding = F.normalize(self.binder_embedder(binder_esm_embedding.mean(dim=1)))

        # get binder and protein embeddings, dot together
        antigen_esm_embedding = self.model(input_ids=antigen.squeeze(), output_hidden_states=True)["hidden_states"][-1]
        prot_embedding = F.normalize(self.prot_embedder(antigen_esm_embedding.mean(dim=1)))

        # embeddings = (B, Clip Embedding Dim)
        logits = torch.matmul(binder_embedding, prot_embedding.T)

        batch_size = batch[0][0].shape[0]
        labels = torch.arange(batch_size).to(self.device)  # diagonal clip loss

        # loss of predicting target using binder
        partner_prediction_loss = F.cross_entropy(logits, labels)

        # loss of predicting binder using target
        binder_prediction_loss = F.cross_entropy(logits.T, labels)

        loss = (partner_prediction_loss + binder_prediction_loss) / 2

        return loss, logits

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
