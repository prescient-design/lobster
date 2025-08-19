import importlib.resources
import random
from collections.abc import Callable

import lightning.pytorch as pl
import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaConfig, LlamaForCausalLM, get_scheduler, pipeline

from lobster.constants import SchedulerType
from lobster.tokenization import PmlmTokenizer, PmlmTokenizerTransform, SmilesTokenizerFast
from lobster.transforms import Transform

from ._clm_configuration import PCLM_CONFIG_ARGS

# Import RDKit and atomic_datasets for validity checking
try:
    from rdkit import Chem
    from atomic_datasets.utils.rdkit import is_molecule_sane
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Validity checking will be disabled.")


class LobsterPCLM2(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "CLM_mini",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1000,
        transform_fn: Callable | Transform | None = None,
        tokenizer_dir: str | None = "pmlm_tokenizer",
        ckpt_path: str = None,
        max_length: int = 512,
        num_key_value_heads: int = None,
        attention_bias: bool = False,
        scheduler: SchedulerType = "constant_with_warmup",
        model_kwargs: dict = None,
        scheduler_kwargs: dict = None,
        num_validation_generations: int = 100,  # Changed from validation_fraction
    ):
        """
        Prescient Protein Causal Language Model.

        Parameters
        ----------
        model_name: one of PCLM_CONFIG_ARGS.keys()
        num_key_value_heads: This is the number of key_value heads that should be used to implement
            Grouped Query Attention. If`num_key_value_heads=num_attention_heads`, the model will
            use Multi Head Attention (MHA), if `num_key_value_heads=1 the model will use
            Multi Query Attention (MQA) otherwise GQA is used.
        scheduler: str, optional
            The type of learning rate scheduler.
        model_kwargs: dict, optional
            Additional keyword arguments to pass to the model.
        scheduler_kwargs: dict, optional
            Additional keyword arguments to pass to the scheduler.
        num_validation_generations: int, optional
            Number of molecules to generate for validity checking (default: 100).

        """
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._ckpt_path = ckpt_path
        self._tokenizer_dir = tokenizer_dir
        self._max_length = max_length
        self._attention_bias = attention_bias
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self._num_validity_checks = num_validation_generations  # Store number of validity checks
        model_kwargs = model_kwargs or {}

        if self._tokenizer_dir is not None:
            path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
            #self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False) #SmilesTokenizerFast(padding="max_length", truncation=True, max_length=self._max_length) #
            self._transform_fn = transform_fn #or PmlmTokenizerTransform(
            #     path,
            #     padding="max_length",
            #     truncation=True,
            #     max_length=self._max_length,
            #     mlm=False,
            # )
            self.tokenizer = self._transform_fn.tokenizer # what did from_pretrained do? 

        config_args = PCLM_CONFIG_ARGS[model_name]
        if num_key_value_heads is None:
            num_key_value_heads = config_args["num_attention_heads"]
        self._num_key_value_heads = num_key_value_heads

        config = LlamaConfig(
            **config_args,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=len(self.tokenizer.get_vocab()),
            max_position_embeddings=self._max_length,
            num_key_value_heads=self._num_key_value_heads,
            attention_bias=self._attention_bias,
            **model_kwargs,
        )
        self.model = LlamaForCausalLM(config)
        self.config = self.model.config

        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        loss, _logits = self._compute_loss(batch)
        ppl = torch.exp(loss)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, _logits = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)

        return {"val_loss": loss}


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

    def _compute_loss(self, batch):
        """
        this is how loss is computed internally i think
        https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/gpt2/modeling_gpt2.py#L1043
        logits = output["logits"]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        PmlmTokenizerTransform prepares the labels, loss computation happens in model.forward()
        """

        if "labels" in batch.keys():            
            output = self.model(
                input_ids=batch["input_ids"].squeeze(1),
                labels=batch["labels"].squeeze(1), #removing this entirely to see what happens
                attention_mask=batch["attention_mask"].squeeze(1),
            )
        else:   
            # bit inconsistent to do this here vs in the datamodule
            
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100 # new based on docs https://huggingface.co/docs/transformers/en/model_doc/llama#transformers.LlamaForCausalLM

            # why are we squeezing?
            # something goes wrong?

            output = self.model(
                input_ids=batch["input_ids"].squeeze(1),
                labels = labels.squeeze(1), 
                attention_mask=batch["attention_mask"].squeeze(1),
            )
        loss = output["loss"]
        logits = output["logits"]

        return loss, logits

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
        preds = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = preds["hidden_states"]  # hidden reps (B, L, H)

        hidden_states = torch.stack(hidden_states, dim=1)  # (B, num layers, L, H)

        return hidden_states

    def sample(
        self,
        seed_seq: str = "M",
        max_length: int = 100,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 950,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
    ):
        # if wanted to avoid adding start and end tokens: self.tokenizer._tokenizer.post_processor = None 
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        
        outseqs = generator(
            seed_seq,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # "this is an awkward expression to step into" -sam
        outseqs = [samp["generated_text"].replace("\n", "") for samp in outseqs]

        return outseqs

    def sequences_to_log_likelihoods(self, sequences: list[str]) -> torch.Tensor:
        outputs = [self.get_nll_and_logits(s) for s in sequences]
        nlls = torch.stack([o[0] for o in outputs])
        return -nlls

    def batch_to_log_likelihoods(self, batch) -> torch.Tensor:
        """
        Adapted from https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/gpt2/modeling_gpt2.py#L1043
        tok_seqs shape: (B, L) of tokenized sequences
        """
        tok_seqs = batch["input_ids"].squeeze(1).to(self.device)
        BZ, _ = tok_seqs.shape
        with torch.no_grad():
            outputs = self.model(input_ids=tok_seqs, labels=tok_seqs)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = tok_seqs[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            loss_per_tok = loss.view(BZ, -1)

            # Compute loss over non-padding tokens but includes CLS and EOS
            mask = shift_labels != self.tokenizer.pad_token_id

        return -1 * torch.where(mask, loss_per_tok, torch.zeros_like(loss_per_tok)).sum(axis=1) / mask.sum(axis=1)

    def get_nll_and_logits(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(self.tokenizer(sequence)["input_ids"], device=self.device).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(input_ids, labels=input_ids)
        nll, logits = outputs.loss, outputs.logits  # (B, L, V), includes CLS and EOS

        return nll, logits

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self._transform_fn(sequences)])
        with torch.inference_mode():
            hidden_states = self.model(input_ids=input_ids, output_hidden_states=True)["hidden_states"]  # [-1]

        return hidden_states

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
