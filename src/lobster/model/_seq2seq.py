import re
from typing import Callable, Union

import lightning.pytorch as pl
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    T5Config,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup

from lobster.transforms import Transform


class PrescientPT5(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "ProstT5",
        is_training: bool = False,
        is_encoder_decoder: bool = True,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1_000,
        freeze: bool = False,
        mask_percentage: float = 0.15,
        transform_fn: Union[Callable, Transform, None] = None,
        config: Union[PretrainedConfig, T5Config, None] = None,
        ckpt_path: str = None,
    ):
        """
        Prescient Protein T5 Model.

        Parameters
        ----------
        model_name: pre-trained model (e.g. ProstT5, prot_t5_xl_uniref50) or name for config (e.g. T5_small)
        is_training: if true, conditional generation model for teacher forcing training
        is_encoder_decoder: if true, full seq2seq arch for translation, else encoder only for embedding
        lr: learning rate
        freeze: freeze all layers except LM head (decoder)
        aho_antibody: if true, log per-region perplexity
        transform_fn: defines tokenizer transform
        config: huggingface config for instantiating a model if ``model_name`` is not specified

        """
        super().__init__()
        self._lr = lr
        self._is_training = is_training
        self._is_encoder_decoder = is_encoder_decoder
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._freeze = freeze
        self._mask_percentage = mask_percentage
        self._ckpt_path = ckpt_path
        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self.tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False)

        if is_training:
            self.model = T5ForConditionalGeneration.from_pretrained(f"Rostlab/{model_name}")
        elif is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"Rostlab/{model_name}")
        else:
            self.model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}")
        self.model.train()

        self.config = self.model.config
        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)

        return {"val_loss": loss}

    def _compute_loss(self, batch):
        output = self.model(
            input_ids=batch["input_ids"].squeeze(1),
            labels=batch["label_ids"].squeeze(1),
            attention_mask=batch["attention_mask"].squeeze(1),
        )

        loss = output["loss"]

        return loss

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

    def sequences_to_latents(self, sequences: list[str]) -> list[torch.Tensor]:
        """Given list of AA or 3Di sequences, return list of all hidden states"""
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]  # replace UNK AAs
        sequences = [
            "<AA2fold>" + " " + s
            if s.isupper()
            else "<fold2AA>" + " " + s  # upper case AAs or lower case 3Di, add whitespace
            for s in sequences
        ]
        ids = self.tokenizer.batch_encode_plus(
            sequences, add_special_tokens=True, padding="longest", return_tensors="pt"
        )

        # hidden states start with <AA2fold> or <fold2AA> special tok and end with padding toks
        with torch.inference_mode():
            hidden_states = self.model(ids.input_ids, attention_mask=ids.attention_mask, output_hidden_states=True)[
                "hidden_states"
            ]  # 25 layers, (B, L, D)

        return hidden_states

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
