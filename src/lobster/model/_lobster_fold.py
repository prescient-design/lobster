import os
import tempfile
from typing import Any, Callable, Dict, Optional, Union

import lightning.pytorch as pl
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.configuration_utils import PretrainedConfig
from transformers.optimization import get_linear_schedule_with_warmup

from lobster.extern.openfold_utils import atom14_to_atom37, backbone_loss
from lobster.transforms import AutoTokenizerTransform, Transform

from ._lobster_fold_base import PPLMFoldBase
from ._lobster_fold_configuration import PPLMFOLD_CONFIG_ARGS, PPLMFoldConfig


class LobsterPLMFold(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "esmfold_v1",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1_000,
        freeze: bool = False,
        mask_percentage: float = 0.15,
        transform_fn: Union[Callable, Transform, None] = None,
        config: Union[PretrainedConfig, None] = None,
        ckpt_path: str = None,
        tokenizer_dir: Optional[str] = "pmlm_tokenizer",
        max_length: int = 512,
        cache_dir: str = None,
    ):
        """
        Prescient Protein Language Model for Folding.

        Parameters
        ----------
        model_name: pre-trained ESM model (e.g. esm2_t6_8M_UR50D) or name for config (e.g. MLM_small)
        lr: learning rate
        freeze: freeze all layers except LM head (decoder)
        aho_antibody: if true, log per-region perplexity
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
        self._ckpt_path = ckpt_path
        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        self._tokenizer_dir = tokenizer_dir
        self._max_length = max_length

        cache_dir = cache_dir or "~/.cache/huggingface/datasets"
        self._cache_dir = cache_dir

        # if "esmfold" in model_name:
        self.tokenizer = AutoTokenizer.from_pretrained(
            # f"facebook/{model_name}",
            "facebook/esmfold_v1"
            # cache_dir=self._cache_dir
        )
        self._transform_fn = AutoTokenizerTransform(
            # f"facebook/{model_name}",
            "facebook/esmfold_v1",
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
        )
        # elif self._tokenizer_dir is not None:
        #     path = importlib.resources.files("lobster") / "assets" / self._tokenizer_dir
        #     self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
        #     self._transform_fn = transform_fn or PmlmTokenizerTransform(
        #         path, padding="max_length", truncation=True, max_length=self._max_length
        #     )

        if model_name and "esmfold" in model_name:
            self.model = EsmForProteinFolding.from_pretrained(
                f"facebook/{model_name}",
                #   cache_dir=self._cache_dir
            )
            if self._freeze:
                self.model.eval()
        # TODO: implement training for scratch with config
        elif model_name and "PPLM" in model_name:
            config_args = PPLMFOLD_CONFIG_ARGS[model_name]
            config = PPLMFoldConfig(
                attention_probs_dropout_prob=0.0,
                mask_token_id=self.tokenizer.mask_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                position_embedding_type="rotary",
                vocab_size=len(self.tokenizer.get_vocab()),
                max_position_embeddings=self._max_length,
                **config_args,
            )
            self.model = PPLMFoldBase(config)
        else:
            self.model = PPLMFoldBase(config)

        self.config = self.model.config
        # from .openfold_utils import make_default_alphafold_loss
        # self.loss_fn = make_default_alphafold_loss()
        self.loss_fn = backbone_loss
        # if self._continue_training and self._continue_checkpoint is not None:
        #     torch.load(self._continue_checkpoint)
        self.save_hyperparameters(logger=False)

    def training_step(self, batch, batch_idx):
        # loss, *logging_dicts = self._compute_loss(batch)
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # loss, *logging_dicts = self._compute_loss(batch)
        loss = self._compute_loss(batch)
        ppl = torch.exp(loss)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_perplexity", ppl, sync_dist=True)

        return {"val_loss": loss}

    def forward_pass(self, batch: Dict[str, Any]):
        outputs = {}
        sequences = batch["sequence"]
        tokenized_output = self.tokenizer.batch_encode_plus(
            sequences,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            tokenized_output["input_ids"].to(self.device),
            tokenized_output["attention_mask"].to(self.device),
        )

        # TODO(amyxlu): potentially resolve position ID for multimers
        # i.e. if linkers are added, this should be translated to Huggingface API format
        structure = self.model(input_ids, attention_mask)
        outputs["sm"] = structure
        outputs["final_atom_positions"] = atom14_to_atom37(outputs["sm"]["positions"][-1], batch)
        outputs["final_atom_mask"] = batch["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        return outputs

    def _compute_loss(self, batch):
        outputs = self.forward_pass(batch)
        # loss, loss_breakdown = self.loss_fn(outputs, batch, _return_breakdown=True)
        return self.loss_fn(
            backbone_rigid_tensor=batch["backbone_rigid_tensor"],
            backbone_rigid_mask=batch["backbone_rigid_mask"],
            traj=outputs["sm"]["frames"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._num_warmup_steps,
            num_training_steps=self._num_training_steps,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FoldseekTransform(Transform):
    """
    Transforms a structure (PDB) into a discretized 3Di sequence.

    Returns
    -------
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (AA_seq, 3Di_struc_seq, combined_seq).

    """

    def __init__(
        self,
        foldseek: str,
        lobster_fold_model_name: str = "esmfold_v1",
        linker_length: int = 25,
    ):
        super().__init__()
        assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
        self._foldseek = foldseek
        self._lobster_fold_model_name = lobster_fold_model_name
        self._linker_length = linker_length
        self._model = LobsterPLMFold(model_name=self._lobster_fold_model_name)
        self._model.eval()
        self._model.model.trunk.set_chunk_size(64)

    def transform(self, sequences: list[str], chains: list = None) -> dict:
        pdb_file = self._lobster_fold_transform(sequences)
        with (
            tempfile.NamedTemporaryFile(delete=True, suffix=".pdb") as pdb_temp_file,
            tempfile.NamedTemporaryFile(delete=True, suffix=".tsv") as tsv_temp_file,
        ):
            with open(pdb_temp_file.name, "w") as f:
                f.write(pdb_file)

            path_to_pdb = pdb_temp_file.name

            cmd = f"{self._foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path_to_pdb} {tsv_temp_file.name}"
            os.system(cmd)

            seq_dict = {}
            name = os.path.basename(path_to_pdb)
            with open(tsv_temp_file.name, "r") as file_handle:
                for _i, line in enumerate(file_handle):
                    # print(line)
                    desc, seq, struc_seq = line.split("\t")[:3]

                    name_chain = desc.split(" ")[0]
                    chain = name_chain.replace(name, "").split("_")[-1]

                    if chains is None or chain in chains:
                        if chain not in seq_dict:
                            combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                            seq_dict[chain] = (seq, struc_seq, combined_seq)

        return seq_dict

    def _lobster_fold_transform(self, sequences: list[str]) -> str:
        # TODO: currently only supports monomer and dimer
        if len(sequences) > 1:  # dimer
            linker = "G" * self._linker_length
            sequence = f"{linker}".join(sequences)
            print(sequence)
        else:  # monomer
            sequence = sequences[0]
        tokenized_input = self._model.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)["input_ids"]

        # add a large offset to the position IDs of the second chain
        if len(sequences) > 1:
            with torch.no_grad():
                position_ids = torch.arange(len(sequence), dtype=torch.long)
                position_ids[len(sequence) + len(linker) :] += 512
                tokenized_input["position_ids"] = position_ids.unsqueeze(0)
                output = self._model.model(**tokenized_input)

                # remove the poly-G linker from the output, so we can display the structure as fully independent chains
                linker_mask = torch.tensor([1] * len(sequence) + [0] * len(linker) + [1] * len(sequence))[None, :, None]
                # output['atom37_atom_exists'] = output['atom37_atom_exists'] * linker_mask.to(output['atom37_atom_exists'].device)
                output["atom37_atom_exists"] = output["atom37_atom_exists"] * linker_mask
        else:
            with torch.no_grad():
                output = self._model.model(tokenized_input)

        pdb_file = self._model.model.output_to_pdb(output)[0]

        return pdb_file

    def validate(self, flat_inputs: list[Any]) -> None:
        pass
