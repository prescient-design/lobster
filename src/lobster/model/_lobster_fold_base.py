"""Adapted from https://github.com/huggingface/transformers/tree/v4.23.1/src/transformers/models"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.esm.modeling_esmfold import (
    EsmFoldingTrunk,
    EsmModel,
    OFProtein,
    atom14_to_atom37,
    categorical_lddt,
    collate_dense_tensors,
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
    residue_constants,
    to_pdb,
)

from ._lobster_fold_configuration import PPLMFoldConfig


class PPLMFoldBasePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PPLMFoldConfig
    base_model_prefix = "PPLMFoldBase"

    # Copied from transformers.models.LMBase.modeling_LMBase.LMBasePreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def update_keys_to_ignore(self, config, del_keys_to_ignore):
        """Remove some keys from ignore list"""
        if not config.tie_word_embeddings:
            # must make a new list, or the class variable gets modified!
            self._keys_to_ignore_on_save = [k for k in self._keys_to_ignore_on_save if k not in del_keys_to_ignore]
            self._keys_to_ignore_on_load_missing = [
                k for k in self._keys_to_ignore_on_load_missing if k not in del_keys_to_ignore
            ]


class PPLMFoldBase(PPLMFoldBasePreTrainedModel):
    _no_split_modules = [
        "EsmFoldStructureModule",
        "EsmFoldTriangularSelfAttentionBlock",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.distogram_bins = 64

        self.esm = EsmModel(config, add_pooling_layer=False)

        self.esm.requires_grad_(False)
        if self.config.fold_config.fp16:
            self.esm.half()

        self.esm_feats = self.config.hidden_size
        self.esm_attns = self.config.num_hidden_layers * self.config.num_attention_heads
        self.esm_layers = self.config.num_hidden_layers
        self.register_buffer("af2_to_esm", self._af2_to_esm_from_vocab_list(config.vocab_list))
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm_layers + 1))

        trunk_config = self.config.fold_config.trunk
        c_s = trunk_config.sequence_state_dim
        c_z = trunk_config.pairwise_state_dim
        self.esm_s_mlp = nn.Sequential(
            nn.LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 0 is padding, N is unknown residues, N + 1 is mask.
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.esm_dict_cls_idx = self.config.vocab_list.index("<cls>")
        self.esm_dict_mask_idx = self.config.vocab_list.index("<mask>")
        self.esm_dict_eos_idx = self.config.vocab_list.index("<eos>")
        self.esm_dict_padding_idx = self.config.vocab_list.index("<pad>")
        if self.config.fold_config.embed_aa:
            self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        self.trunk = EsmFoldingTrunk(trunk_config)

        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        self.lddt_bins = 50
        structure_module_config = trunk_config.structure_module
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(structure_module_config.sequence_dim),
            nn.Linear(
                structure_module_config.sequence_dim,
                self.config.fold_config.lddt_head_hid_dim,
            ),
            nn.Linear(
                self.config.fold_config.lddt_head_hid_dim,
                self.config.fold_config.lddt_head_hid_dim,
            ),
            nn.Linear(self.config.fold_config.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    def _af2_to_esm_from_vocab_list(vocab_list: List[str]) -> torch.Tensor:
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [vocab_list.index("<pad>")] + [vocab_list.index(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        masking_pattern: Optional[torch.Tensor] = None,
        num_recycles: Optional[int] = None,
    ) -> Dict[str, Any]:
        r"""
        Returns:
        -------

        Example:
        -------
        ```python
        >>> from transformers import AutoTokenizer, EsmForProteinFolding

        >>> model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        >>> inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt", add_special_tokens=False)  # A tiny random peptide
        >>> outputs = model(**inputs)
        >>> folded_positions = outputs.positions
        ```

        """
        cfg = self.config.fold_config

        aa = input_ids  # B x L
        B = aa.shape[0]
        L = aa.shape[1]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(aa, device=device)
        if position_ids is None:
            position_ids = torch.arange(L, device=device).expand_as(input_ids)

        # === ESM ===
        esmaa = self.af2_idx_to_esm_idx(aa, attention_mask)

        if masking_pattern is not None:
            masked_aa, esmaa, mlm_targets = self.bert_mask(aa, esmaa, attention_mask, masking_pattern)
        else:
            masked_aa = aa
            mlm_targets = None

        # We get sequence and pair representations from whatever version of ESM /
        # configuration we are using. The sequence representation esm_s is always
        # present. The pair embedding esm_z may be present depending on the
        # configuration of the model. If esm_z is not used by the model then it
        # is returned as None here.
        esm_s = self.compute_language_model_representations(esmaa)

        # Convert esm_s and esm_z, if present, to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)

        if cfg.ablate_sequence:
            esm_s = esm_s * 0

        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)

        s_z_0 = s_s_0.new_zeros(B, L, L, cfg.trunk.pairwise_state_dim)

        if self.config.fold_config.embed_aa:
            s_s_0 += self.embedding(masked_aa)

        structure: dict = self.trunk(s_s_0, s_z_0, aa, position_ids, attention_mask, no_recycles=num_recycles)
        # Documenting what we expect:
        structure = {
            k: v
            for k, v in structure.items()
            if k
            in [
                "s_z",
                "s_s",
                "frames",
                "sidechain_frames",
                "unnormalized_angles",
                "angles",
                "positions",
                "states",
            ]
        }

        # Add BERT mask for the loss to use, if available.
        if mlm_targets:
            structure["mlm_targets"] = mlm_targets

        disto_logits = self.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits

        structure["aatype"] = aa
        make_atom14_masks(structure)
        # Of course, this doesn't respect the true mask because it doesn't know about it...
        # We're not going to properly mask change of index tensors:
        #    "residx_atom14_to_atom37",
        #    "residx_atom37_to_atom14",
        for k in [
            "atom14_atom_exists",
            "atom37_atom_exists",
        ]:
            structure[k] *= attention_mask.unsqueeze(-1)
        structure["residue_index"] = position_ids

        lddt_head = self.lddt_head(structure["states"]).reshape(structure["states"].shape[0], B, L, -1, self.lddt_bins)
        structure["lddt_head"] = lddt_head
        plddt = categorical_lddt(lddt_head[-1], bins=self.lddt_bins)
        structure["plddt"] = plddt

        ptm_logits = self.ptm_head(structure["s_z"])
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = compute_tm(ptm_logits, max_bin=31, no_bins=self.distogram_bins)
        structure.update(compute_predicted_aligned_error(ptm_logits, max_bin=31, no_bins=self.distogram_bins))
        return structure

    def af2_idx_to_esm_idx(self, aa, mask):
        # avoid indexing on different devices
        if self.af2_to_esm.device != aa.device:
            self.af2_to_esm = self.af2_to_esm.to(aa.device)
        aa = (aa + 1).masked_fill(mask != 1, 0)
        return self.af2_to_esm[aa]

    def compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        B, L = esmaa.shape  # B = batch size, L = sequence length.

        if self.config.fold_config.bypass_lm:
            esm_s = torch.zeros(B, L, self.esm_s_combine.size[0], -1, self.esm_feats, device=device)
            return esm_s

        bosi, eosi = self.esm_dict_cls_idx, self.esm_dict_eos_idx
        bos = esmaa.new_full((B, 1), bosi)
        eos = esmaa.new_full((B, 1), self.esm_dict_padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # Use the first padding index as eos during inference.
        esmaa[range(B), (esmaa != 1).sum(1)] = eosi

        # _, esm_z, esm_s = self.esm(esmaa, return_pairs=self.config.fold_config.use_esm_attn_map)
        # Because we do not support use_esm_attn_map in the HF port as it is not used in any public models,
        # esm_z is always None
        esm_hidden_states = self.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)["hidden_states"]
        esm_s = torch.stack(esm_hidden_states, dim=2)

        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C

        return esm_s

    def bert_mask(self, aa, esmaa, mask, pattern):
        new_aa = aa.clone()
        target = aa.clone()
        new_esmaa = esmaa.clone()
        new_aa[pattern == 1] = self.mask_idx
        target[pattern != 1] = 0
        new_esmaa[pattern == 1] = self.esm_dict_mask_idx
        return new_aa, new_esmaa, target

    @torch.no_grad()
    def infer(
        self,
        seqs: Union[str, List[str]],
        position_ids=None,
    ):
        if isinstance(seqs, str):
            lst = [seqs]
        else:
            lst = seqs
        # Returns the raw outputs of the model given an input sequence.
        device = next(self.parameters()).device
        aatype = collate_dense_tensors(
            [
                torch.from_numpy(
                    residue_constants.sequence_to_onehot(
                        sequence=seq,
                        mapping=residue_constants.restype_order_with_x,
                        map_unknown_to_x=True,
                    )
                )
                .to(device)
                .argmax(dim=1)
                for seq in lst
            ]
        )  # B=1 x L
        mask = collate_dense_tensors([aatype.new_ones(len(seq)) for seq in lst])
        position_ids = (
            torch.arange(aatype.shape[1], device=device).expand(len(lst), -1)
            if position_ids is None
            else position_ids.to(device)
        )
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        return self.forward(
            aatype,
            mask,
            position_ids=position_ids,
        )

    @staticmethod
    def output_to_pdb(output: Dict) -> List[str]:
        """Returns the pbd (file) string from the model given the model output."""
        output = {k: v.to("cpu").numpy() for k, v in output.items()}
        pdbs = []
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        final_atom_mask = output["atom37_atom_exists"]
        for i in range(output["aatype"].shape[0]):
            aa = output["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = output["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=output["plddt"][i],
            )
            pdbs.append(to_pdb(pred))
        return pdbs

    def infer_pdb(self, seqs, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        assert isinstance(seqs, str)
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)[0]

    def infer_pdbs(self, seqs: List[str], *args, **kwargs) -> List[str]:
        """Returns the pdb (file) string from the model given an input sequence."""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)
