import importlib.resources
from collections.abc import Callable, Iterable
from typing import Literal

import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from transformers.configuration_utils import PretrainedConfig

from lobster.tokenization import CUSTOM_TOKENIZER, PmlmConceptTokenizerTransform, PmlmTokenizer
from lobster.transforms import Transform

from ._mlm_configuration import PMLM_CONFIG_ARGS, PMLMConfig
from .lm_base import LMBaseForConditionalMaskedLM


class LobsterConditionalPMLM(pl.LightningModule):
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
        use_descriptors: bool = False,
        descriptors_transform: str | list[str] = None,
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

        if self._tokenizer_dir is not None:
            path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
            self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
            self._transform_fn = transform_fn or PmlmConceptTokenizerTransform(
                path, padding="max_length", truncation=True, max_length=self._max_length
            )

            self._concepts_name = self._transform_fn.concepts_name
            self._n_concepts_size = len(self._concepts_name)

        config_args = PMLM_CONFIG_ARGS[model_name]
        config = PMLMConfig(
            attention_probs_dropout_prob=0.0,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            position_embedding_type=self._position_embedding_type,
            vocab_size=len(self.tokenizer.get_vocab()),
            max_position_embeddings=self._max_length,
            **config_args,
        )

        if descriptors_transform is not None:
            self.custom_concept_emb = []
            config.embed_concepts = True
            config.concept_input_size = []
            config.concept_emb_size = []
            for descriptor in descriptors_transform:
                des = CUSTOM_TOKENIZER[descriptor]()
                self._concepts_name.extend(des.concepts_names)
                self._n_concepts_size += des.emd_size
                config.concept_emb_size.append(des.emd_size)
                config.concept_input_size.append(sum(des.concept_size))
        self._n_concepts = len(self._concepts_name)
        config.n_concepts = self._n_concepts
        config.has_conditioning = True
        config.conditioning_type = "pre_encoder"
        config.n_concepts_size = self._n_concepts_size
        self.use_descriptors = use_descriptors

        self.model = LMBaseForConditionalMaskedLM(config)
        if self._initial_mask_percentage is not None:
            assert self._initial_mask_percentage > self._mask_percentage
        self.config = self.model.config
        self.save_hyperparameters(logger=False)

        self.transform_fn_inf = PmlmConceptTokenizerTransform(
            path, padding="max_length", truncation=True, max_length=self._max_length
        )

    def training_step(self, batch, batch_idx):
        loss, *logging_dicts = self._compute_loss(batch)
        # ,auxiliary_loss_dict
        ppl = torch.exp(loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
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
        self.log("val_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)
        if any(logging_dicts):
            logging_dicts = [{f"val/{k}_ppl": v for k, v in d.items()} for d in logging_dicts]
            for d in logging_dicts:
                self.log_dict(d, sync_dist=True)

        return {"val_loss": loss}

    def _compute_loss(self, input):
        # torch.cuda.empty_cache()

        if self.use_descriptors:
            batch, des = input
            concepts_to_emb = [des["all_concepts"]]
        else:
            batch = input
            concepts_to_emb = None
        toks = batch["input_ids"].squeeze(1)
        labels = toks.clone()
        masked_toks = self._mask_inputs(toks)
        labels[masked_toks != self.tokenizer.mask_token_id] = -100  # only calculate loss on masked tokens

        output = self.model(
            input_ids=masked_toks,
            concepts=batch["all_concepts"],
            concepts_to_emb=concepts_to_emb,
            attention_mask=batch["attention_mask"].squeeze(1),
            labels=labels,
        )
        loss = output["loss"]

        logging_dicts = []

        del masked_toks, toks

        return loss, logging_dicts

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self._lr, betas=(self._beta1, self._beta2), eps=self._eps
        )
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def predict_step(self, batch, batch_idx) -> pd.DataFrame:
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

        # Tokenize full sequence
        ref_seq = " ".join(sequence)
        ref_seq_indices = torch.tensor(self.tokenizer.encode(ref_seq)) - 4
        # -4 to align since first tokens are special, BERT-related.

        # Generate full sequence variants with aa's masked one by one, tokenize
        seqs_masked = [" ".join([aa if j != i else "<mask>" for j, aa in enumerate(sequence)]) for i in range(N)]
        seqs_mask_encoded = torch.tensor(
            [self.tokenizer.encode(masked_seq) for masked_seq in seqs_masked], device=self.device
        )

        with torch.inference_mode():
            logits = torch.vstack(
                [
                    self.model(input_ids=toks)["logits"]
                    for toks in torch.tensor_split(seqs_mask_encoded, N // batch_size)
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
            logits = self.model.lm_head(x)
        tokens = [self.tokenizer.decode(logit.argmax(dim=-1)) for logit in logits]
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        tokens = [t.replace(" ", "") for t in tokens]
        tokens = ["".join([t for t in seq if t in aa_toks]) for seq in tokens]
        return tokens

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self.transform_fn_inf(sequences)])

        concepts = torch.concat(
            [toks["all_concepts"].unsqueeze(0).to(self.device) for toks in self.transform_fn_inf(sequences)]
        )

        if self.use_descriptors:
            concepts_to_emb = [torch.zeros(concepts.shape[0], self.config.concept_input_size[0]).to(self.device)]
        else:
            concepts_to_emb = None
        with torch.inference_mode():
            hidden_states = self.model(
                input_ids=input_ids,
                concepts=concepts,
                concepts_to_emb=concepts_to_emb,
                inference=True,
                output_hidden_states=True,
            )["hidden_states"]  # [-1]

        return hidden_states

    def _perturb_seq(self, sequences: list[str], sigma: float = 5.0) -> list[str]:
        h = self.sequences_to_latents(sequences)
        h_perturbed = h + torch.randn(h.shape) * sigma * h.var()
        sequences = self.latent_embeddings_to_sequences(h_perturbed)

        return sequences

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
