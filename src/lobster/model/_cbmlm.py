import importlib.resources
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Union

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup

from lobster.tokenization import CUSTOM_TOKENIZER, PmlmConceptTokenizerTransform, PmlmTokenizer

from ._mlm_configuration import PMLM_CONFIG_ARGS, PMLMConfig
from .lm_base import LMBaseForConditionalMaskedLM


class LobsterCBMPMLM(pl.LightningModule):
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
        initial_mask_percentage: Optional[float] = None,
        transform_fn: Union[Callable, None] = None,
        config: Union[PretrainedConfig, None] = None,
        ckpt_path: str = None,
        tokenizer_dir: Optional[str] = "pmlm_tokenizer",
        max_length: int = 512,
        max_num_concepts: int = 2000,
        position_embedding_type: Literal["rotary", "absolute"] = "rotary",
        concept_hp: float = 0.1,
        orthogonality_hp: float = 0.1,
        use_descriptors: bool = False,
        add_embedding_noise: bool = False,
        noise_mean: float = 0.0,
        noise_std_min: float = 0.0,
        noise_std_max: float = 0.0,
        descriptors_transform: Union[str, list[str]] = None,
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
        self.concept_hp = concept_hp
        self._position_embedding_type = position_embedding_type
        self.orthogonality_hp = orthogonality_hp
        self._max_num_concepts = max_num_concepts
        self._add_embedding_noise = add_embedding_noise
        self._noise_mean = noise_mean
        self._noise_std_min = noise_std_min
        self._noise_std_max = noise_std_max

        load_pretrained = config is None and model_name not in PMLM_CONFIG_ARGS

        # setup tokenizer
        if load_pretrained:
            assert model_name is not None

            # first check if this exists locally
            concepts_path = Path(model_name).resolve() / "concepts.json"
            if not concepts_path.exists():
                concepts_path = hf_hub_download(model_name, "concepts.json")

            with open(concepts_path, "r") as f:
                concepts = json.load(f)

            self.tokenizer = PmlmTokenizer.from_pretrained(model_name, do_lower_case=False)
            self._transform_fn = transform_fn or PmlmConceptTokenizerTransform(
                model_name,
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
                concepts_name=concepts,
            )
            self.transform_fn_inf = transform_fn or PmlmConceptTokenizerTransform(
                model_name,
                padding="max_length",
                truncation=True,
                max_length=self._max_length,
                concepts_name=concepts,
            )
            self._concepts_name = self._transform_fn.concepts_name
            self._n_concepts = len(self._concepts_name)
            self._seq_concepts = len(self._concepts_name)
        else:
            assert self._tokenizer_dir is not None
            path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
            self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
            self._transform_fn = transform_fn or PmlmConceptTokenizerTransform(
                path, padding="max_length", truncation=True, max_length=self._max_length
            )
            self.transform_fn_inf = PmlmConceptTokenizerTransform(
                path, padding="max_length", truncation=True, max_length=self._max_length
            )
            self._concepts_name = self._transform_fn.concepts_name
            self._n_concepts = len(self._concepts_name)
            self._seq_concepts = len(self._concepts_name)

        if descriptors_transform is not None:
            for descriptor in descriptors_transform:
                des = CUSTOM_TOKENIZER[descriptor]()
                self._concepts_name.extend(des.concepts_names_full)
                self._n_concepts += len(des.concepts_names_full)
                self._descriptor_highlevel_concepts = des.concepts_names

        if load_pretrained:
            assert model_name is not None
            self.model = LMBaseForConditionalMaskedLM.from_pretrained(model_name)
        else:
            if model_name is not None:
                #                assert config is None, f"Cannot supply both `config` and `model_name` {config=}, {model_name=}"

                config_args = PMLM_CONFIG_ARGS[model_name]
                config = PMLMConfig(
                    attention_probs_dropout_prob=0.0,
                    mask_token_id=self.tokenizer.mask_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    position_embedding_type=self._position_embedding_type,
                    vocab_size=len(self.tokenizer.get_vocab()),
                    max_position_embeddings=self._max_length,
                    tie_word_embeddings=False,
                    **config_args,
                )

            config.num_labels = self._n_concepts
            config.n_concepts = self._n_concepts
            config.concept_emb = 2
            config.has_conditioning = True
            config.conditioning_type = "cbm"
            config.max_num_concepts = self._max_num_concepts
            config.add_embedding_noise = self._add_embedding_noise
            config.noise_mean = self._noise_mean
            config.noise_std_max = self._noise_std_max
            config.noise_std_min = self._noise_std_min

            self.model = LMBaseForConditionalMaskedLM(config)

        self.use_descriptors = use_descriptors
        if self._initial_mask_percentage is not None:
            assert self._initial_mask_percentage > self._mask_percentage

        self.config = self.model.config
        self.save_hyperparameters(logger=False)

    def orthogonality_loss(self, concept_emb, unk_emb):
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        output = torch.abs(cos(concept_emb, unk_emb))
        return output.mean()

    def training_step(self, batch, batch_idx):
        loss, *logging_dicts, auxiliary_loss_dict = self._compute_loss(batch)
        ppl = torch.exp(auxiliary_loss_dict["task_loss"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)
        if any(logging_dicts):
            logging_dicts = [{f"train/{k}_ppl": v for k, v in d.items()} for d in logging_dicts]
            for d in logging_dicts:
                self.log_dict(d, sync_dist=True)

        p_mask = self._get_p_mask()
        self.log("train_p_mask", p_mask, sync_dist=True)

        prefix = "train_"
        cbm_loss_dict = {prefix + key: value for key, value in auxiliary_loss_dict.items()}
        self.log_dict(cbm_loss_dict, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, *logging_dicts, auxiliary_loss_dict = self._compute_loss(batch, validation=True)
        ppl = torch.exp(auxiliary_loss_dict["task_loss"])
        self.log("val_loss", loss, prog_bar=True, on_step=True, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)
        if any(logging_dicts):
            logging_dicts = [{f"val/{k}_ppl": v for k, v in d.items()} for d in logging_dicts]
            for d in logging_dicts:
                self.log_dict(d, sync_dist=True)

        prefix = "val_"
        cbm_loss_dict = {prefix + key: value for key, value in auxiliary_loss_dict.items()}
        self.log_dict(cbm_loss_dict, sync_dist=True)
        return {"val_loss": loss}

    def _compute_loss(self, input, validation=False):
        if self.use_descriptors:
            batch, des = input
            for i in range(len(self._descriptor_highlevel_concepts)):
                concept = des[self._descriptor_highlevel_concepts[i]]
                first_one_mask = concept[:, 0] == 1
                mask = torch.ones_like(concept, dtype=torch.bool)
                mask[:, 0] = ~first_one_mask

                if i == 0:
                    all_masks = mask
                    all_concepts = concept
                else:
                    all_masks = torch.concat([all_masks, mask], dim=1)
                    all_concepts = torch.concat([all_concepts, concept], dim=1)

            mask = torch.ones_like(batch["all_concepts"], dtype=torch.bool)
            all_masks = torch.concat([mask, all_masks], dim=1)
            all_concepts = torch.concat([batch["all_concepts"], all_concepts], dim=1)
        else:
            batch = input
            all_masks = torch.ones_like(batch["all_concepts"], dtype=torch.bool)
            all_concepts = batch["all_concepts"]

        toks = batch["input_ids"].squeeze(1)
        labels = toks.clone()
        masked_toks = self._mask_inputs(toks)
        labels[masked_toks != self.tokenizer.mask_token_id] = -100  # only calculate loss on masked tokens

        if validation:
            output = self.model(
                input_ids=masked_toks,
                concepts=all_concepts,
                attention_mask=batch["attention_mask"].squeeze(1),
                inference=True,
                labels=labels,
            )
        else:
            output = self.model(
                input_ids=masked_toks,
                concepts=all_concepts,
                attention_mask=batch["attention_mask"].squeeze(1),
                labels=labels,
            )

        pred_concept = output["concepts"]
        task_loss = output["loss"]

        masked_pred_concept = pred_concept[all_masks]
        masked_all_concepts = all_concepts[all_masks]
        overall_concept_loss = self._n_concepts * F.mse_loss(masked_pred_concept, masked_all_concepts, reduction="mean")

        auxiliary_loss_dict = {
            "task_loss": task_loss,
            "orth_loss": self.orthogonality_loss(output["cbm_emd"], output["unk_emd"]),
            "overall_concept_loss": overall_concept_loss,
        }

        for c in range(self._seq_concepts):
            concept_loss = F.mse_loss(pred_concept[:, c].unsqueeze(-1), batch[self._concepts_name[c]])
            auxiliary_loss_dict[self._concepts_name[c]] = concept_loss

        logging_dicts = []
        loss = (
            task_loss
            + self.concept_hp * overall_concept_loss
            + self.orthogonality_hp * auxiliary_loss_dict["orth_loss"]
        )
        return loss, logging_dicts, auxiliary_loss_dict

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

    def predict_step(self, batch, batch_idx) -> pd.DataFrame:
        toks = batch["input_ids"].squeeze()
        toks = toks.to(self.device)
        attention_mask = batch["attention_mask"].squeeze().to(self.device)
        with torch.inference_mode():
            preds = self.model(input_ids=toks, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = preds["hidden_states"][-1]  # last layer hidden reps (B, L, H)

        df = pd.DataFrame(
            hidden_states.mean(dim=1).cpu(),
            columns=[f"embedding_{idx}" for idx in range(hidden_states.shape[-1])],
        )
        return df

    def naturalness(self, sequences: Iterable[str]) -> torch.Tensor:
        out = [self._naturalness_single_sequence(seq, batch_size=32) for seq in sequences]
        return torch.tensor(out)

    def _naturalness_single_sequence(
        self,
        sequence: str,
        batch_size: int = 32,
        return_probs: bool = False,
    ) -> tuple[float, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        N = len(sequence)

        ref_seq = " ".join(sequence)
        ref_seq_indices = torch.tensor(self.tokenizer.encode(ref_seq)) - 4

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

        raw_log_probs = torch.nn.functional.log_softmax(logits[:, 1:-1, 4:24], dim=-1)[
            torch.arange(N), torch.arange(N), :
        ]
        sum_log_probs = raw_log_probs[torch.arange(N), ref_seq_indices[1:-1]].sum()

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

    def _mask_inputs(self, train_inputs: torch.Tensor, mask_arr=None, p_mask=None):
        if mask_arr is None:
            rand = torch.rand(train_inputs.shape, device=train_inputs.device)
            if p_mask is None:
                p_mask = self._get_p_mask()

            mask_arr = (
                (rand < p_mask)
                * (train_inputs != self.tokenizer.cls_token_id)
                * (train_inputs != self.tokenizer.pad_token_id)
                * (train_inputs != self.tokenizer.eos_token_id)
            )
        else:
            mask_arr = (
                mask_arr
                * (train_inputs != self.tokenizer.cls_token_id)
                * (train_inputs != self.tokenizer.pad_token_id)
                * (train_inputs != self.tokenizer.eos_token_id)
            )

        selection = [torch.flatten(mask_arr[i].nonzero()).tolist() for i in range(train_inputs.shape[0])]
        masked_inputs = train_inputs.clone()
        for i in range(train_inputs.shape[0]):
            masked_inputs[i, selection[i]] = self.tokenizer.mask_token_id

        return masked_inputs

    def _freeze_all_but_lm_head(self):
        for name, param in self.model.named_parameters():
            if "lm_head" not in name:
                param.requires_grad = False

    def latent_embeddings_to_sequences(self, x: torch.Tensor) -> list[str]:
        with torch.inference_mode():
            logits = self.model.lm_head(x)
        tokens = [self.tokenizer.decode(logit.argmax(dim=-1)) for logit in logits]
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        tokens = [t.replace(" ", "") for t in tokens]
        tokens = ["".join([t for t in seq if t in aa_toks]) for seq in tokens]
        return tokens

    def _create_intervene_mask(self, tensor, num_features, num_features_to_zero, intervention_type="positive"):
        masks = []
        for i in range(len(tensor)):
            # only sort values for actual amino acids, ignores start token, end token and padding
            if intervention_type == "positive":
                sorted_tensor, sorted_inds = torch.sort(tensor[i, 1 : num_features[i] + 1], dim=0, descending=False)
            elif intervention_type == "negative":
                sorted_tensor, sorted_inds = torch.sort(tensor[i, 1 : num_features[i] + 1], dim=0, descending=True)
            else:
                raise ValueError(f"'{intervention_type}' unknown")
            sorted_inds = sorted_inds + 1  # correct for removing start token before
            mask = torch.zeros_like(tensor[i], dtype=torch.bool)
            mask[sorted_inds[:num_features_to_zero]] = True
            masks.append(mask)
        mask = torch.stack(masks, dim=0)
        return mask

    def intervene_on_sequences(
        self, sequences: list[str], concept: str, edits: int, intervention_type: str
    ) -> list[str]:
        self.eval()
        try:
            concept_index = self.transform_fn_inf.concepts_name.index(concept)
            print(f" Intervening on {concept} with index: {concept_index}")
        except ValueError:
            print(f"'{concept}' is not in the list")

        ########## tokenize

        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self.transform_fn_inf(sequences)])
        attention_mask = torch.concat(
            [toks["attention_mask"].to(self.device) for toks in self.transform_fn_inf(sequences)]
        )

        ####### get feature attribution
        forward_output = self.model(
            input_ids=input_ids, inference=True, attention_mask=attention_mask, requires_grad=True
        )

        pred_concepts_value = forward_output["concepts"]
        concept_ = pred_concepts_value[:, concept_index]
        input_ = forward_output["input_emb"]
        attribution = torch.autograd.grad(torch.unbind(concept_), input_, allow_unused=True)[0]

        ####### creating mask
        mask_emb = self.model.LMBase.embeddings.word_embeddings.weight[-1].detach()
        attribution = torch.sum(attribution * (input_ - mask_emb), dim=2)

        num_features = torch.sum(
            (input_ids != self.tokenizer.cls_token_id)
            * (input_ids != self.tokenizer.pad_token_id)
            * (input_ids != self.tokenizer.eos_token_id),
            dim=1,
        ).to(self.device)
        mask = self._create_intervene_mask(attribution, num_features, edits, intervention_type)

        masked_toks = self._mask_inputs(input_ids, mask_arr=mask)

        concept_mask = torch.zeros(input_ids.shape[0], self._n_concepts).to(self.device)
        concept_mask[:, concept_index] = 1
        new_concepts_value = pred_concepts_value.clone()

        ########## intervening on the concepts
        if intervention_type == "positive":
            new_concepts_value[:, concept_index] = 1
        else:
            new_concepts_value[:, concept_index] = 0

        intervene_value = (concept_mask, new_concepts_value)
        logits_masked = self.model(
            input_ids=masked_toks, inference=True, intervene=intervene_value, attention_mask=attention_mask
        )["logits"]
        logits_masked = logits_masked.detach()

        ########## transform the seqence
        aa_toks = list("ARNDCEQGHILKMFPSTWYV")
        pred_masked_tokens = []
        for j, logit in enumerate(logits_masked):
            pred_tok = logit.argmax(dim=-1)
            mask = masked_toks[j].eq(self.tokenizer.mask_token_id).int()
            pred_masked_token = (input_ids[j] * (1 - mask)) + (pred_tok * mask)
            pred_masked_token = self.tokenizer.decode(pred_masked_token)
            pred_masked_token = pred_masked_token.replace(" ", "")
            pred_masked_token = "".join([t for t in pred_masked_token if t in aa_toks])
            pred_masked_tokens.append(pred_masked_token)
        return pred_masked_tokens

    def list_supported_concept(self):
        return self.transform_fn_inf.concepts_name

    def sequences_to_concepts(self, sequences: list[str]) -> torch.Tensor:
        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self.transform_fn_inf(sequences)])
        with torch.inference_mode():
            pred_concepts = self.model(input_ids=input_ids, inference=True)["concepts"]
        return pred_concepts

    def sequences_to_concepts_emb(self, sequences: list[str]) -> torch.Tensor:
        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self.transform_fn_inf(sequences)])
        with torch.inference_mode():
            cbm_emd = self.model(input_ids=input_ids, inference=True)["cbm_emd"]
        return cbm_emd

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self.transform_fn_inf(sequences)])
        with torch.inference_mode():
            hidden_states = self.model(input_ids=input_ids, inference=True, output_hidden_states=True)["hidden_states"]
        return hidden_states

    def _perturb_seq(self, sequences: list[str], sigma: float = 5.0) -> list[str]:
        h = self.sequences_to_latents(sequences)
        h_perturbed = h + torch.randn(h.shape) * sigma * h.var()
        sequences = self.latent_embeddings_to_sequences(h_perturbed)
        return sequences

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], *args, **kwargs):
        self.model.save_pretrained(save_directory, *args, **kwargs)
        self.tokenizer.save_pretrained(save_directory, *args, **kwargs)

        concepts_path = Path(save_directory).resolve() / "concepts.json"
        with open(concepts_path, "w") as f:
            json.dump(self._concepts_name, f)
