import re

import pandas as pd
import torch
from prescient.constants import (
    FR_RANGES_AHO,
    VERNIER_ZONES,  # Apparently these are 1 indexed?
)

## Tokenize the germline sequences and inputs in place
## since the tokenization is different depending on what you want to do...


class GraftGermline:
    def __init__(
        self,
        germline_df,
        tokenization_transform,
        keep_vernier_from_animal=None,
        keep_const_idxs_dict=None,
    ):
        """TODO: add docstrings"""
        self.germline_df = germline_df
        self.tokenization_transform = tokenization_transform
        self.id_to_token = tokenization_transform._auto_tokenizer._id_to_token
        self.keep_vernier_from_animal = keep_vernier_from_animal
        if keep_const_idxs_dict is None:
            keep_const_idxs_dict = {"H": [], "L": []}
        self.keep_const_idxs_dict = keep_const_idxs_dict
        (
            self.h_idxs_to_graft,
            self.l_idxs_to_graft,
            self._constant_idxs_dict,
        ) = self.get_aho_fr_idxs()

        self.germline_df["fv_heavy_aho_tok"] = [
            t["input_ids"].squeeze()
            for t in self.tokenization_transform(
                self.germline_df["fv_heavy_aho"].tolist()
            )
        ]
        self.germline_df["fv_light_aho_tok"] = [
            t["input_ids"].squeeze()
            for t in self.tokenization_transform(
                self.germline_df["fv_light_aho"].tolist()
            )
        ]

    def get_aho_fr_idxs(self):
        h_idxs_to_graft = []
        l_idxs_to_graft = []

        for k, v in FR_RANGES_AHO.items():
            # Add 1 to account for start token
            if k.startswith("H"):
                start_idx, end_idx = v
                h_idxs_to_graft.extend(range(start_idx + 1, end_idx + 1))
            elif k.startswith("L"):
                start_idx, end_idx = v
                l_idxs_to_graft.extend(range(start_idx + 1, end_idx + 1))
            constant_idxs_dict = {"H": [], "L": []}
        if self.keep_vernier_from_animal is not None:
            assert self.keep_vernier_from_animal in VERNIER_ZONES.keys()
            vernier_dict = VERNIER_ZONES[self.keep_vernier_from_animal]
            constant_idxs_dict = {
                "H": [int(idx) for idx in vernier_dict["H"]],
                "L": [int(idx) for idx in vernier_dict["L"]],
            }
        for k, v in self.keep_const_idxs_dict.items():
            constant_idxs_dict[k].extend(v)

        print(
            "Keeping additional non-CDR indices constant \n",
            "heavy: ",
            constant_idxs_dict["H"],
            "\n light:",
            constant_idxs_dict["L"],
        )
        h_idxs_to_graft = list(set(h_idxs_to_graft) - set(constant_idxs_dict["H"]))
        l_idxs_to_graft = list(set(l_idxs_to_graft) - set(constant_idxs_dict["L"]))

        return h_idxs_to_graft, l_idxs_to_graft, constant_idxs_dict

    def _graft_germline_sequences(
        self, toks_to_graft: torch.Tensor, gl_toks: torch.Tensor, idxs_to_graft: list
    ) -> torch.Tensor:
        """
        Returns germline FWs grafted onto the tokenized input sequence with respect to each chain
        """
        # pass in only 1 sequence to graft at a time
        assert toks_to_graft.shape[0] == 1

        toks_to_graft = toks_to_graft.repeat((gl_toks.shape[0], 1))
        toks_to_graft[:, idxs_to_graft] = gl_toks[:, idxs_to_graft]
        return toks_to_graft

    def graft_fv_heavy_germline_sequences(
        self, fv_heavy_seq_toks: torch.Tensor
    ) -> torch.Tensor:
        gl_fv_heavy_aho_tok = torch.stack(
            self.germline_df["fv_heavy_aho_tok"].to_list()
        )
        return self._graft_germline_sequences(
            fv_heavy_seq_toks, gl_fv_heavy_aho_tok, self.h_idxs_to_graft
        )

    def graft_fv_light_germline_sequences(
        self, fv_light_seq_toks: torch.Tensor
    ) -> torch.Tensor:
        gl_fv_light_aho_tok = torch.stack(
            self.germline_df["fv_light_aho_tok"].to_list()
        )
        return self._graft_germline_sequences(
            fv_light_seq_toks, gl_fv_light_aho_tok, self.l_idxs_to_graft
        )

    def _convert_toks_to_string(self, seq_toks: torch.Tensor) -> list[str]:
        seqs = []
        for row in seq_toks:
            raw_str = "".join([self.id_to_token[int(i)] for i in row])
            match = re.search("<cls>(.*)<eos>", raw_str)
            seqs.append(match.group(1) if match else "")
        return seqs

    def process_grafted_seq_toks(
        self,
        grafted_fv_heavy_aho_toks: torch.Tensor,
        grafted_fv_light_aho_toks: torch.Tensor,
    ) -> pd.DataFrame:
        germline_df_ = self.germline_df[
            [
                "fv_heavy_aho",
                "fv_light_aho",
                "HFR1",
                "HFR2",
                "HFR3a",
                "HFR3b",
                "HFR4",
                "H1",
                "H2",
                "H3",
                "H4",
                "LFR1",
                "LFR2",
                "LFR3a",
                "LFR3b",
                "LFR4",
                "L1",
                "L2",
                "L3",
                "L4",
                "heavy_v_gene",
                "heavy_j_gene",
                "light_v_gene",
                "light_j_gene",
            ]
        ]

        germline_df_.rename(
            columns={col: col + "_germline" for col in germline_df_.columns},
            inplace=True,
        )

        seqs_df_ = pd.DataFrame(
            {
                "fv_heavy_aho": self._convert_toks_to_string(grafted_fv_heavy_aho_toks),
                "fv_light_aho": self._convert_toks_to_string(grafted_fv_light_aho_toks),
            }
        )

        return pd.concat([germline_df_, seqs_df_], axis=1)

    def graft(
        self, fv_heavy_aho: str, fv_light_aho: str
    ) -> (pd.DataFrame, torch.Tensor, torch.Tensor):
        fv_heavy_aho_tok = self.tokenization_transform(fv_heavy_aho)["input_ids"]
        fv_light_aho_tok = self.tokenization_transform(fv_light_aho)["input_ids"]

        grafted_fv_heavy_aho_toks = self.graft_fv_heavy_germline_sequences(
            fv_heavy_aho_tok
        )
        grafted_fv_light_aho_toks = self.graft_fv_light_germline_sequences(
            fv_light_aho_tok
        )

        return (
            self.process_grafted_seq_toks(
                grafted_fv_heavy_aho_toks, grafted_fv_light_aho_toks
            ),
            grafted_fv_heavy_aho_toks,
            grafted_fv_light_aho_toks,
        )
