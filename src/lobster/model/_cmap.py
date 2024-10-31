import random
from typing import Optional

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import BCELoss
from torchmetrics import Precision, Recall

from lobster.data import ESM_MODEL_NAMES, PMLM_MODEL_NAMES
from lobster.model import LMBaseContactPredictionHead, LobsterPMLM
from lobster.model._esm import ESMBase


class ContactPredictionHead(pl.LightningModule):
    def __init__(
        self,
        model_name: Optional[str] = None,
        checkpoint: Optional[str] = None,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        seed: int = 0,
        lr: float = 1e-3,
        ckpt_path: str = None,
    ):
        """
        Contact Prediction Head, following the same format as ESM2
        """
        super().__init__()
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._lr = lr
        self._seed = seed

        # Set random seeds
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        print(f"------ Loading checkpoint: {checkpoint}")

        if model_name in PMLM_MODEL_NAMES:
            base_model = LobsterPMLM(model_name=model_name)
            base_model.load_from_checkpoint(checkpoint)

        elif model_name in ESM_MODEL_NAMES:
            base_model = ESMBase(model_name=model_name)

        else:
            raise ValueError(
                f"Model name: {model_name} is not supported, please choose from {PMLM_MODEL_NAMES + ESM_MODEL_NAMES}"
            )

        self.model_name = model_name
        self.checkpoint = checkpoint
        self.base_model = base_model
        for _name, param in self.base_model.named_parameters():  # freeze pre-trained encoder
            param.requires_grad = False

        if model_name in ESM_MODEL_NAMES:
            hidden_size = self.base_model.model.embed_dim
            self.contact_head = LMBaseContactPredictionHead(
                in_features=self.base_model.model.num_layers * self.base_model.model.attention_heads
            )
        else:
            hidden_size = self.base_model.model.config.hidden_size
            self.contact_head = LMBaseContactPredictionHead(
                in_features=self.base_model.config.num_hidden_layers * self.base_model.config.num_attention_heads
            )

        self.hidden_size = hidden_size

        # metrics
        # TODO - insert P@L & P@L/5 --> precision of top L hits where L is length of protein
        # TODO - insert overall precision?

        # TODO - correct input features, etc.

        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.test_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")
        self.test_recall = Recall(task="binary")

        self.loss = (
            BCELoss()
        )  # output of predict_contacts is after sigmoid, so using BCELoss instead of BCEWithLogitsLoss

        self.save_hyperparameters(logger=False)

    def predict_contacts(
        self,
        tokens: Tensor,
        attentions: Tensor,
        attention_mask: Tensor,
    ):
        """
        TODO - should this be the forward method? Or just make it predict_contacts?
        Replicates the function of models._lm_base.LMBaseModelRelative.predict_contacts, but made available
        for all PLM types
        """
        attns = torch.stack(attentions, dim=1)  # Matches the original model layout
        # In the original model, attentions for padding tokens are completely zeroed out.
        # This makes no difference most of the time because the other tokens won't attend to them,
        # but it does for the contact prediction task, which takes attentions as input,
        # so we have to mimic that here.
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        return self.contact_head(tokens, attns)

    def training_step(self, batch, batch_idx):
        preds, targets = self._compute_loss(batch)
        loss = self.loss(preds, targets)  # TODO - add L1 regularization on the outputs?
        try:
            precision = self.train_precision(preds, targets)
            recall = self.train_recall(preds, targets)
        except IndexError:
            print(f"IndexError in training_step {batch_idx}, setting precision to 0.0")
            precision = torch.tensor(0.0)
            recall = torch.tensor(0.0)
        loss_dict = {"train/precision": precision, "train/recall": recall, "train/bce": loss}

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.inference_mode():
            preds, targets = self._compute_loss(batch)
            loss = self.loss(preds, targets)
            try:
                precision = self.val_precision(preds, targets)
                recall = self.val_recall(preds, targets)
            except IndexError:
                print(f"IndexError in validation_step {batch_idx}, setting precision to 0.0")
                precision = torch.tensor(0.0)
                recall = torch.tensor(0.0)
            loss_dict = {"val/precision": precision, "val/recall": recall, "val/bce": loss}

            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log_dict(loss_dict)
            return loss

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            preds, targets = self._compute_loss(batch)
            loss = self.loss(preds, targets)
            try:
                precision = self.test_precision(preds, targets)
                recall = self.val_recall(preds, targets)
            except IndexError:
                print(f"IndexError in test_step {batch_idx}, setting precision to 0.0")

                precision = torch.tensor(0.0)
                recall = torch.tensor(0.0)

            loss_dict = {"test/precision": precision, "test/recall": recall, "test/bce": loss}

            self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log_dict(loss_dict)
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr, betas=(self._beta1, self._beta2), eps=self._eps)
        # scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        # return [optimizer], [scheduler]
        return optimizer

    def _compute_loss(self, batch: dict):
        """
        Expecting bach as the output of ESMBatchConverterPPI:
        {
            "tokens1": tokens1,
            "tokens2": tokens2,
            "attention_mask1": attention_mask_1,
            "attention_mask2": attention_mask_2,
            "contact_map": contact_map,
        }
        """
        tokens1, tokens2 = batch["tokens1"], batch["tokens2"]
        attention_mask1, attention_mask2 = batch["attention_mask1"], batch["attention_mask2"]
        contact_map = batch["contact_map"]

        # Concat sequences if traditional model, have two inputs if relative rep
        # TODO - debug & test this

        tokens_cat = torch.cat([tokens1, tokens2], dim=1)
        attention_mask_all = torch.cat([attention_mask1, attention_mask2], dim=1)

        if self.model_name in PMLM_MODEL_NAMES:
            # Assume one set of positional encodings for the concatenated token
            inpt = {"input_ids": tokens_cat, "attention_mask": attention_mask_all}
            attns = self.base_model.model(**inpt, return_dict=True, output_attentions=True).attentions

        if self.model_name in ESM_MODEL_NAMES:
            # output = self.base_model.model(tokens_cat, need_head_weights=True) # TODO - test this
            # attns = tuple(output["attentions"])
            # pred_map = self.predict_contacts(
            #     tokens=tokens_cat, attentions=attns, attention_mask=attention_mask_all
            # )
            # return pred_map, 'pred map'
            # Alternatively could do
            pred_map = self.base_model.model(tokens_cat, need_head_weights=True, return_contacts=True)["contacts"]

        else:
            pred_map = self.predict_contacts(tokens=tokens_cat, attentions=attns, attention_mask=attention_mask_all)

        # TODO - handle different batch sizes
        # Note the next part is designed for batch_size = 1
        length1 = len(batch["tokens1"][0]) - 2
        length2 = len(batch["tokens2"][0]) - 2

        pred_map = pred_map[:, :length1, -length2:].squeeze()
        true_map = contact_map[0].squeeze()

        # in case pred had to be truncated, crop the true map to match
        if pred_map.shape != true_map.shape:
            min_x = min(pred_map.shape[0], true_map.shape[0])
            min_y = min(pred_map.shape[1], true_map.shape[1])
            true_map = true_map[:min_x, :min_y]

        return pred_map, true_map
        # return {"predicted_contact_map": pred_map,
        #         "full_output": output,
        #         "precision" : precision,
        #         }

        # Pass sequences & attention maps through ESMContact
        # TODO - finish & test this
        # P&L computation

        # return loss, concat_maps, out
