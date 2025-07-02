import copy
import importlib.resources
import os
from collections.abc import Callable, Iterable
from typing import Literal

import lightning.pytorch as pl
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, EsmForMaskedLM, get_scheduler
from transformers.configuration_utils import PretrainedConfig

from lobster.constants import SchedulerType
from lobster.tokenization import PmlmTokenizer, PmlmTokenizerTransform
from lobster.transforms import AutoTokenizerTransform, Transform

from ._mlm_configuration import PMLM_CONFIG_ARGS, PMLMConfig
from .lm_base import LMBaseForMaskedLM


class LobsterPMLM(pl.LightningModule):
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
        initial_mask_percentage: float | None = None,
        transform_fn: Callable | Transform | None = None,
        config: PretrainedConfig | None = None,
        ckpt_path: str = None,
        tokenizer_dir: str | None = "pmlm_tokenizer",
        max_length: int = 512,
        position_embedding_type: Literal["rotary", "absolute"] = "rotary",
        use_bfloat16: bool = False,
        scheduler: SchedulerType = "constant_with_warmup",
        model_kwargs: dict = None,
        scheduler_kwargs: dict = None,
    ):
        """
        Prescient Protein Masked Language Model.

        Parameters
        ----------
        model_name: pre-trained ESM model (e.g. esm2_t6_8M_UR50D) or name for config (e.g. MLM_small)
        lr: learning rate
        freeze: freeze all layers except LM head (decoder)
        mask_percentage: final masking rate
        initial_mask_percentage: initial masking rate, if not None, linear dynamic mask rate
            scheduler will be used. initial should be greater than final.
        transform_fn: defines tokenizer transform
        config: huggingface config for instantiating a model if ``model_name`` is not specified
        tokenizer_dir: a tokenizer saved to src/lobster/assets
        max_length: max sequence length the model will see
        use_bfloat16: use bfloat16 instead of float32 for ESM-C model weights
        scheduler: str, optional
            The type of learning rate scheduler.
        model_kwargs: dict, optional
            Additional keyword arguments to pass to the model.
        scheduler_kwargs: dict, optional
            Additional keyword arguments to pass to the scheduler.

        """
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._freeze = freeze
        self._mask_percentage = mask_percentage
        self._initial_mask_percentage = initial_mask_percentage
        self._ckpt_path = ckpt_path
        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._tokenizer_dir = tokenizer_dir
        self._max_length = max_length
        self._position_embedding_type = position_embedding_type
        self._use_esmc = False
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        model_kwargs = model_kwargs or {}

        load_pretrained = config is None and model_name not in PMLM_CONFIG_ARGS

        # setup tokenizer
        if load_pretrained:
            assert model_name is not None

            # TODO remove special handling for esm models
            if "esm2" in model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}", do_lower_case=False)
                self._transform_fn = AutoTokenizerTransform(
                    f"facebook/{model_name}",
                    padding="max_length",
                    truncation=True,
                    max_length=self._max_length,
                )
            elif model_name == "esmc":
                if not use_bfloat16:
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        "Synthyra/ESMplusplus_small", trust_remote_code=True
                    )
                else:
                    self.model = AutoModelForMaskedLM.from_pretrained(
                        "Synthyra/ESMplusplus_small", trust_remote_code=True, torch_dtype=torch.bfloat16
                    )
                self._use_esmc = True
            else:
                self.tokenizer = PmlmTokenizer.from_pretrained(model_name, do_lower_case=False)
                self._transform_fn = PmlmTokenizerTransform(
                    model_name,
                    padding="max_length",
                    truncation=True,
                    max_length=self._max_length,
                )
        else:
            assert self._tokenizer_dir is not None
            path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
            self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
            self._transform_fn = transform_fn or PmlmTokenizerTransform(
                path, padding="max_length", truncation=True, max_length=self._max_length
            )

        # setup model
        if load_pretrained:
            assert model_name is not None

            # TODO remove special handling for esm models
            if "esm2" in model_name:
                self.model = EsmForMaskedLM.from_pretrained(f"facebook/{model_name}")
            elif model_name == "esmc":
                self.tokenizer = self.model.tokenizer
            else:
                self.model = LMBaseForMaskedLM.from_pretrained(model_name)
        else:
            if model_name is not None:
                # use named config
                # FIXME this assert fails for some pretrained checkpoints
                #                assert config is None, "Cannot supply both `config` and `model_name`"
                assert model_name in PMLM_CONFIG_ARGS
                config_args = PMLM_CONFIG_ARGS[model_name]
                config = PMLMConfig(
                    **config_args,
                    attention_probs_dropout_prob=0.0,
                    mask_token_id=self.tokenizer.mask_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    position_embedding_type=self._position_embedding_type,
                    vocab_size=len(self.tokenizer.get_vocab()),
                    max_position_embeddings=self._max_length,
                    **model_kwargs,
                )
            self.model = LMBaseForMaskedLM(config)

        if self._initial_mask_percentage is not None:
            assert self._initial_mask_percentage > self._mask_percentage

        if self._freeze:
            self._freeze_all_but_lm_head()

        self.config = self.model.config
        # if self._continue_training and self._continue_checkpoint is not None:
        #     torch.load(self._continue_checkpoint)
        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        loss, *logging_dicts = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)
        if any(logging_dicts):
            logging_dicts = [{f"train/{k}_ppl": v for k, v in d.items()} for d in logging_dicts]
            for d in logging_dicts:
                self.log_dict(d, sync_dist=True)

        p_mask = self._get_p_mask()
        self.log("train_p_mask", p_mask, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, *logging_dicts = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)
        if any(logging_dicts):
            logging_dicts = [{f"val/{k}_ppl": v for k, v in d.items()} for d in logging_dicts]
            for d in logging_dicts:
                self.log_dict(d, sync_dist=True)

        return {"val_loss": loss}

    def _compute_loss(self, batch):
        # torch.cuda.empty_cache()
        batch, _targets = batch  # targets are the FASTA IDs
        toks = batch["input_ids"].squeeze(1)
        labels = toks.clone()
        masked_toks = self._mask_inputs(toks)
        labels[masked_toks != self.tokenizer.mask_token_id] = -100  # only calculate loss on masked tokens

        output = self.model(
            input_ids=masked_toks,
            attention_mask=batch["attention_mask"].squeeze(1),
            labels=labels,
        )
        loss = output["loss"]
        # loss = F.cross_entropy(output["logits"].permute(0, 2, 1), toks)  # (B, V, L)

        logging_dicts = []

        del masked_toks, toks

        return loss, logging_dicts

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model for ONNX compiling.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input tensor.
        attention_mask: torch.Tensor
            The attention mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.

        """
        preds = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = preds["hidden_states"]  # hidden reps (B, L, H)

        hidden_states = torch.stack(hidden_states, dim=1)  # (B, num layers, L, H)

        return hidden_states

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )
        scheduler = get_scheduler(
            self.scheduler,
            optimizer,
            num_training_steps=self._num_training_steps,
            num_warmup_steps=self._num_warmup_steps,
            scheduler_specific_kwargs=self.scheduler_kwargs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def predict_step(self, batch, batch_idx) -> pd.DataFrame:
        # batch, _targets = batch  # targets are the FASTA IDs
        toks = batch["input_ids"].squeeze()
        toks = toks.to(self.device)
        attention_mask = batch["attention_mask"].squeeze().to(self.device)
        with torch.inference_mode():
            preds = self.model(input_ids=toks, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = preds["hidden_states"][-1]  # last layer hidden reps (B, L, H)

        # mean pool over AAs
        df = pd.DataFrame(
            hidden_states.mean(dim=1).cpu(),
            columns=[f"embedding_{idx}" for idx in range(hidden_states.shape[-1])],
        )

        return df

    def embed_dataset(
        self, sequences: list[str], batch_size: int = 32, residue_wise: bool = True, num_workers: int = 0
    ) -> dict[str, torch.Tensor]:
        if not self._use_esmc:
            raise NotImplementedError("This method is only implemented for ESM-C models.")
        embeddings = self.model.embed_dataset(
            sequences=sequences,  # list of protein strings
            batch_size=batch_size,  # embedding batch size
            max_len=self._max_length,  # truncate to max_len
            full_embeddings=residue_wise,  # return residue-wise embeddings
            full_precision=False,  # store as float32
            pooling_type="mean",  # use mean pooling if protein-wise embeddings
            num_workers=0,  # data loading num workers
            sql=False,  # return dictionary of sequences and embeddings
        )
        return embeddings

    def naturalness(self, sequences: Iterable[str]) -> torch.Tensor:
        out = [
            self._naturalness_single_sequence(
                seq,
                batch_size=32,
            )
            for seq in sequences
        ]

        return torch.tensor(out)

    def _naturalness_single_sequence(
        self,
        sequence: str,
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> tuple[float, tuple[torch.Tensor, torch.Tensor] | None]:
        N = len(sequence)

        if self._use_esmc:
            # Tokenize full sequence
            encoded_seq = self.tokenizer.encode(sequence)
            ref_seq_indices = torch.tensor(encoded_seq) - 4
            # -4 to align since first tokens are special, BERT-related.

            # Generate full sequence variants with aa's masked one by one, tokenize
            seqs_masked = [copy.deepcopy(encoded_seq) for x in range(N)]
            for i in range(N):
                seqs_masked[i][i + 1] = self.tokenizer.mask_token_id
            seqs_mask_encoded = torch.tensor(seqs_masked, device=self.device)
        else:
            # Tokenize full sequence
            ref_seq = " ".join(sequence)
            ref_seq_indices = torch.tensor(self.tokenizer.encode(ref_seq)) - 4
            # -4 to align since first tokens are special, BERT-related.

            # Generate full sequence variants with aa's masked one by one, tokenize
            seqs_masked = [" ".join([aa if j != i else "<mask>" for j, aa in enumerate(sequence)]) for i in range(N)]
            seqs_mask_encoded = torch.tensor(
                [self.tokenizer.encode(masked_seq) for masked_seq in seqs_masked],
                device=self.device,
            )

        if N < batch_size:
            batch_size_ = N
        else:
            batch_size_ = batch_size
        with torch.inference_mode():
            logits = torch.vstack(
                [
                    self.model(input_ids=toks.to(self.device))["logits"]
                    for toks in torch.tensor_split(seqs_mask_encoded, N // batch_size_)
                ]
            )

        # raw_log_probs [N, 20]: log probability for each WT amino acid
        raw_log_probs = torch.nn.functional.log_softmax(logits[:, 1:-1, 4:24], dim=-1)[
            torch.arange(N), torch.arange(N), :
        ]
        # sum of log probabilities that the model assigns to the true amino acid in each masked position
        sum_log_probs = raw_log_probs[torch.arange(N), ref_seq_indices[1:-1]].sum()  # chop off bos/eos

        naturalness_score = (1.0 / torch.exp(-sum_log_probs / N)).item()

        if return_probs:
            return naturalness_score, (raw_log_probs, ref_seq_indices[1:-1].detach())
        else:
            return naturalness_score

    def _get_p_mask(self):
        if self._initial_mask_percentage is not None:
            p_mask = self._initial_mask_percentage + (self.trainer.global_step / self._num_training_steps) * (
                self._mask_percentage - self._initial_mask_percentage
            )
        else:
            p_mask = self._mask_percentage

        return p_mask

    def _mask_inputs(self, train_inputs: torch.Tensor, p_mask=None):
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(train_inputs.shape, device=train_inputs.device)
        if p_mask is None:
            p_mask = self._get_p_mask()
        # create mask array
        mask_arr = (
            (rand < p_mask)
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
        with torch.inference_mode():
            if self._use_esmc:
                logits = self.model.sequence_head(x)
            else:
                logits = self.model.lm_head(x)
        tokens = [self.tokenizer.decode(logit.argmax(dim=-1)) for logit in logits]
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        tokens = [t.replace(" ", "") for t in tokens]
        tokens = ["".join([t for t in seq if t in aa_toks]) for seq in tokens]
        return tokens

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        if self._use_esmc:
            input_ids = self.tokenizer(
                sequences, padding="max_length", truncation=True, max_length=self._max_length, return_tensors="pt"
            )
            input_ids = input_ids["input_ids"].to(self.device)
        else:
            input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self._transform_fn(sequences)])
        with torch.inference_mode():
            hidden_states = self.model(input_ids=input_ids, output_hidden_states=True)["hidden_states"]  # [-1]
        return hidden_states

    def _perturb_seq(self, sequences: list[str], sigma: float = 5.0) -> list[str]:
        h = self.sequences_to_latents(sequences)
        h_perturbed = h + torch.randn(h.shape) * sigma * h.var()
        sequences = self.latent_embeddings_to_sequences(h_perturbed)

        return sequences

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_pretrained(self, save_directory: str | os.PathLike, *args, **kwargs):
        self.model.save_pretrained(save_directory, *args, **kwargs)
        self.tokenizer.save_pretrained(save_directory, *args, **kwargs)
