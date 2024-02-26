import importlib.resources
import random
from typing import Callable, Optional, TypeVar, Union

import lightning.pytorch as pl
import torch
from yeji.transforms import Transform
from torch.nn import MSELoss
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from lobster.tokenization import PmlmTokenizer, PmlmTokenizerTransform

from ._mlm_configuration import PMLMConfig
from ._rlm_configuration import RLM_CONFIG_ARGS
from .lm_base import LMBaseForMaskedLMRelative

T = TypeVar("T")


class PrescientPRLM(pl.LightningModule):
    def __init__(
        self,
        model_name: str = None,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1_000,
        freeze: bool = False,
        mask_percentage: float = 0.15,
        continue_training: bool = False,
        transform_fn: Union[Callable, Transform, None] = None,
        ckpt_path: str = None,
        tokenizer_dir: Optional[str] = "pmlm_tokenizer",
        max_length: int = 512,
        rel_hyper: float = 1,
        rec_hyper: float = 1,
        abs_hyper: float = 1,
        exc_hyper: float = 1,
        abs_const: float = 1,
        collate_fn: Optional[Callable[[list[T]], T]] = None,
    ):
        """
        Prescient Protein Masked Language Model.

        Parameters
        ----------
        model_name: pre-trained  model
        lr: learning rate
        freeze: freeze all layers except LM head

        """

        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._freeze = freeze
        self._mask_percentage = mask_percentage
        self._continue_training = continue_training
        self._ckpt_path = ckpt_path
        self._rel_hyper = rel_hyper
        self._abs_hyper = abs_hyper
        self._exc_hyper = exc_hyper
        self._rec_hyper = rec_hyper

        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._tokenizer_dir = tokenizer_dir
        self._max_length = max_length

        if "esm2" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"facebook/{model_name}", do_lower_case=False
            )
        elif self._tokenizer_dir is not None:
            path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
            self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
            self._transform_fn = transform_fn or PmlmTokenizerTransform(
                path, padding="max_length", truncation=True, max_length=self._max_length
            )

        self.null_token = self.tokenizer._convert_token_to_id("<null_1>")

        # rlm_config = RLM_CONFIG_ARGS[model_name]
        config_args = RLM_CONFIG_ARGS[model_name]

        # self.config = PMLMConfig(
        #         # num_labels=rlm_config["num_labels"],
        #         problem_type=rlm_config["problem_type"],
        #         num_hidden_layers=rlm_config["num_hidden_layers"],
        #         num_attention_heads=rlm_config["num_attention_heads"],
        #         intermediate_size=rlm_config["intermediate_size"],
        #         hidden_size=rlm_config["hidden_size"],
        #         attention_probs_dropout_prob=rlm_config["attention_probs_dropout_prob"],
        #         mask_token_id=rlm_config["mask_token_id"],
        #         pad_token_id=rlm_config["pad_token_id"],
        #         token_dropout=rlm_config["token_dropout"],
        #         position_embedding_type=rlm_config["position_embedding_type"],
        #         vocab_size=rlm_config["vocab_size"],
        #         layer_norm_eps=rlm_config["layer_norm_eps"])

        self.config = PMLMConfig(
            attention_probs_dropout_prob=0.0,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            position_embedding_type="rotary",
            vocab_size=len(self.tokenizer.get_vocab()),
            **config_args,
        )

        self.model = LMBaseForMaskedLMRelative(self.config)

        if self._continue_training and self._ckpt_path is not None:
            self.model = torch.load(self._ckpt_path)
        self.save_hyperparameters(logger=False)
        self.criterion = MSELoss()
        self.abs_const = abs_const

    def training_step(self, batch, batch_idx):
        loss, loss_dicts = self._compute_loss(batch)
        ppl = torch.exp(loss_dicts["rec"])
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)
        if any(loss_dicts):
            for k, v in loss_dicts.items():
                self.log("train_" + k, v, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, loss_dicts = self._compute_loss(batch)
        ppl = torch.exp(loss_dicts["rec"])
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)
        if any(loss_dicts):
            for k, v in loss_dicts.items():
                self.log("val_" + k, v, sync_dist=True)

        return {"val_loss": loss}

    def forward(self, batch):
        torch.cuda.empty_cache()
        toks_i, toks_j = batch
        toks_i = toks_i["input_ids"].squeeze(1)
        toks_j = toks_j["input_ids"].squeeze(1)

        null_toks = self._null_inputs(toks_i)
        masked_toks_i = self._mask_inputs(toks_i)
        _, prediction_tokens = self.model(input_ids_a=masked_toks_i, input_ids_b=null_toks)

        output_i_j, _ = self.model(input_ids_a=toks_i, input_ids_b=toks_j)

        output_i_null, _ = self.model(input_ids_a=toks_i, input_ids_b=null_toks)

        output_null_j, _ = self.model(input_ids_a=null_toks, input_ids_b=toks_j)

        return output_i_j, output_i_null, output_null_j, prediction_tokens

    def _compute_loss(self, batch):
        # TODO - replace first part of this with call to self.forward?
        torch.cuda.empty_cache()
        toks_i_batch, toks_j_batch = batch
        toks_i = toks_i_batch["input_ids"].squeeze(1)
        toks_j = toks_j_batch["input_ids"].squeeze(1)
        null_toks = self._null_inputs(toks_i)

        ### Reconstruction Loss ###

        # Randomly choose an input to mask
        # Either mask fist input if rand = 1
        # Mask second input of rand =2
        # Mask third input if rand =3

        rand = random.randint(1, 3)
        if rand == 1:

            masked_toks_i = self._mask_inputs(toks_i)
            labels_i = toks_i.clone()

            labels_i[masked_toks_i != self.tokenizer.mask_token_id]
            output = self.model(
                input_ids_a=null_toks,
                input_ids_b=masked_toks_i,
                attention_mask_b=toks_i_batch["attention_mask"].squeeze(1),
                labels=labels_i,
                mask_task=True,
                null_input=True,
            )

        elif rand == 2:

            masked_toks_j = self._mask_inputs(toks_j)
            labels_j = toks_j.clone()

            labels_j[masked_toks_j != self.tokenizer.mask_token_id]
            output = self.model(
                input_ids_a=null_toks,
                input_ids_b=masked_toks_j,
                attention_mask_b=toks_j_batch["attention_mask"].squeeze(1),
                labels=labels_j,
                mask_task=True,
                null_input=True,
            )

        else:

            masked_toks_i = self._mask_inputs(toks_i)
            masked_toks_j = self._mask_inputs(toks_j)
            labels_i = toks_i.clone()
            labels_j = toks_j.clone()
            labels_i[masked_toks_i != self.tokenizer.mask_token_id]

            labels_j[masked_toks_j != self.tokenizer.mask_token_id]

            labels = torch.cat((labels_i, labels_j), 1)

            output = self.model(
                input_ids_a=masked_toks_i,
                attention_mask_a=toks_i_batch["attention_mask"].squeeze(1),
                input_ids_b=masked_toks_j,
                attention_mask_b=toks_j_batch["attention_mask"].squeeze(1),
                labels=labels,
                mask_task=True,
                null_input=None,
            )

        logging_dicts = {}

        logging_dicts["rec"] = output["loss"]

        ### Relative Representation Loss ###

        relaltive_i_j = self.model(
            input_ids_a=toks_i,
            attention_mask_a=toks_i_batch["attention_mask"].squeeze(1),
            input_ids_b=toks_j,
            attention_mask_b=toks_j_batch["attention_mask"].squeeze(1),
            mask_task=False,
        )

        relaltive_i_null = self.model(
            input_ids_a=toks_i,
            attention_mask_a=toks_i_batch["attention_mask"].squeeze(1),
            input_ids_b=null_toks,
            mask_task=False,
        )

        relaltive_null_i = self.model(
            input_ids_a=null_toks,
            input_ids_b=toks_i,
            attention_mask_b=toks_i_batch["attention_mask"].squeeze(1),
            mask_task=False,
        )

        relaltive_null_j = self.model(
            input_ids_a=null_toks,
            input_ids_b=toks_j,
            attention_mask_b=toks_j_batch["attention_mask"].squeeze(1),
            mask_task=False,
        )

        relative_embedding = relaltive_null_i + relaltive_i_j - relaltive_null_j

        norm_relative_embedding = torch.norm(relative_embedding, dim=1)
        zeros = self._constant_zeros(norm_relative_embedding)

        logging_dicts["rel"] = self.criterion(norm_relative_embedding, zeros)

        # ### Absolute Representation Loss ###
        # rand = random.randint(1,2)
        # if rand==1:
        #     norm_relaltive = torch.norm(relaltive_null_i, dim=1)
        # else:
        #     norm_relaltive = torch.norm(relaltive_null_j, dim=1)

        # constant = self._constant_reg(norm_relaltive)

        # logging_dicts["abs"] = self.criterion(norm_relaltive, constant)

        del output, relaltive_null_i, relaltive_i_j, relaltive_null_j, relaltive_i_null

        loss = (
            self._rel_hyper * logging_dicts["rel"]
            # + self._abs_hyper * logging_dicts["abs"]
            + self._rec_hyper * logging_dicts["rec"]
        )

        return loss, logging_dicts

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self._lr, betas=(self._beta1, self._beta2), eps=self._eps
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._num_training_steps,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _null_inputs(self, train_inputs: torch.Tensor):
        # create an array of null tokens floats with equal dimensions to input_ids tensor
        null_input = torch.ones(
            train_inputs.shape, device=train_inputs.device, dtype=train_inputs.dtype
        )
        null_input = null_input * self.null_token
        null_input[:, 0] = self.tokenizer.cls_token_id
        null_input[:, -1] = self.tokenizer.eos_token_id

        return null_input

    def _constant_zeros(self, train_inputs: torch.Tensor):
        # create an array of null tokens floats with equal dimensions to input_ids tensor
        constant_zero = torch.zeros(
            train_inputs.shape, device=train_inputs.device, dtype=train_inputs.dtype
        )

        return constant_zero

    def _constant_reg(self, train_inputs: torch.Tensor):
        # create an array of null tokens floats with equal dimensions to input_ids tensor
        constant_reg = torch.ones(
            train_inputs.shape, device=train_inputs.device, dtype=train_inputs.dtype
        )
        constant_reg = constant_reg * self.abs_const

        return constant_reg

    def _mask_inputs(self, train_inputs: torch.Tensor):
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(train_inputs.shape, device=train_inputs.device)
        # create mask array
        mask_arr = (
            (rand < self._mask_percentage)
            * (train_inputs != self.tokenizer.cls_token_id)
            * (train_inputs != self.tokenizer.pad_token_id)
            * (train_inputs != self.tokenizer.eos_token_id)
        )  # don't mask cls, pad, eos

        selection = []  # masked token positions

        for i in range(train_inputs.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

        masked_inputs = train_inputs.clone()
        for i in range(train_inputs.shape[0]):
            masked_inputs[i, selection[i]] = self.tokenizer.mask_token_id  # 32

        return masked_inputs

    def _freeze_all_but_lm_head(self):
        for name, param in self.model.named_parameters():
            if "lm_head" not in name:  # Skip the lm head
                param.requires_grad = False

    def latent_embeddings_to_sequences(self, x: torch.Tensor) -> list[str]:
        """x: (B, L, H) size tensor of hidden states"""
        logits = self.model.lm_head(x)
        tokens = [self.tokenizer.decode(logit.argmax(dim=-1)) for logit in logits]
        tokens = [t.replace(" ", "") for t in tokens]
        return tokens

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        """NOTE - this currently works for single protein inputs only"""
        _labels, _strs, toks = self._collate_fn(
            [(str(idx), seq) for idx, seq in enumerate(sequences)]
        )

        toks = toks.to(self.device)
        attn_mask_a = toks != self.tokenizer.pad_token_id  # for outputting attn_mask
        null_toks = self._null_inputs(toks)

        with torch.inference_mode():

            hidden_states = self.model(
                input_ids_a=toks, input_ids_b=null_toks, output_hidden_states=True
            )[0].hidden_states

        # Remove null token embeddings
        hidden_states_a = tuple(
            [
                hidden_states[ix][:, : hidden_states[ix].shape[1] // 2, :]
                for ix in range(len(hidden_states))
            ]
        )

        return hidden_states_a, attn_mask_a

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
