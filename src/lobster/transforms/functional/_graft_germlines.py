import re

import pandas as pd
import torch

from lobster.constants import (
    FR_RANGES_AHO,
    VERNIER_ZONES,  # Apparently these are 1 indexed?
)

## Tokenize the germline sequences and inputs in place
## since the tokenization is different depending on what you want to do...


class GraftGermline:
    def __init__(
        self, germline_df, tokenization_transform, chain: str, keep_const_idxs_list: list, keep_vernier_from_animal=None
    ):
        """
        germline_df: pd.DataFrame
            Dataframe containing germline sequences from IMGT
        tokenization_transform: callable
            Tokenization function to convert sequences to tokenized input
        chain: str
            Chain to graft onto
        keep_const_idxs_list: list
            List of indices to keep constant
        keep_vernier_from_animal: str
            Animal to keep constant regions from
        """
        assert chain in ["H", "L"]
        self.chain = chain
        self.chain_str = "heavy" if chain == "H" else "light"
        self.germline_df = germline_df
        self.tokenization_transform = tokenization_transform
        self.id_to_token = tokenization_transform._auto_tokenizer._id_to_token
        self.keep_vernier_from_animal = keep_vernier_from_animal
        self.keep_const_idxs_list = keep_const_idxs_list
        self.idxs_to_graft = self.get_aho_fr_idxs()

        self.germline_df[f"fv_{self.chain_str}_aho_tok"] = [
            t["input_ids"].squeeze()
            for t in self.tokenization_transform(self.germline_df[f"fv_{self.chain_str}_aho"].tolist())
        ]

    def get_aho_fr_idxs(self):
        idxs_to_graft = []

        for k, v in FR_RANGES_AHO.items():
            # Add 1 to account for start token
            if k.startswith(self.chain):
                start_idx, end_idx = v
                idxs_to_graft.extend(range(start_idx + 1, end_idx + 1))

        if self.keep_vernier_from_animal is not None:
            # Assume murine vernier zones are consistent, if not, add indices to the Prescient constants
            if self.keep_vernier_from_animal == "rat":
                self.keep_vernier_from_animal = "mouse"
            assert self.keep_vernier_from_animal in VERNIER_ZONES.keys()
            vernier_dict = VERNIER_ZONES[self.keep_vernier_from_animal]
            constant_idxs_list = [int(idx) for idx in vernier_dict[self.chain]]
            self.keep_const_idxs_list.extend(constant_idxs_list)

        # print("Keeping additional non-CDR indices constant \n", f"{self.chain_str}: ",  self.keep_const_idxs_list)
        idxs_to_graft = list(set(idxs_to_graft) - set(self.keep_const_idxs_list))

        return idxs_to_graft

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

    def graft_fv_germline_sequences(self, fv_seq_toks: torch.Tensor) -> torch.Tensor:
        gl_fv_aho_tok = torch.stack(self.germline_df[f"fv_{self.chain_str}_aho_tok"].to_list())
        return self._graft_germline_sequences(fv_seq_toks, gl_fv_aho_tok, self.idxs_to_graft)

    def _convert_toks_to_string(self, seq_toks: torch.Tensor) -> list[str]:
        seqs = []
        for row in seq_toks:
            raw_str = "".join([self.id_to_token[int(i)] for i in row])
            match = re.search("<cls>(.*)<eos>", raw_str)
            seqs.append(match.group(1) if match else "")
        return seqs

    def process_grafted_seq_toks(
        self,
        grafted_fv_aho_toks: torch.Tensor,
    ) -> pd.DataFrame:
        col_names = [f"fv_{self.chain_str}_aho", f"{self.chain_str}_v_gene", f"{self.chain_str}_j_gene"]
        col_names.extend(self.germline_df.filter(like=f"{self.chain}").columns)
        germline_df_ = self.germline_df[col_names]

        germline_df_.rename(
            columns={col: col + "_germline" for col in germline_df_.columns},
            inplace=True,
        )

        seqs_df_ = pd.DataFrame(
            {
                f"fv_{self.chain_str}_aho": self._convert_toks_to_string(grafted_fv_aho_toks),
            }
        ).set_axis(germline_df_.index)

        return pd.concat([germline_df_, seqs_df_], axis=1)

    def graft(
        self,
        fv_aho: str,
    ) -> (pd.DataFrame, torch.Tensor):
        fv_aho_tok = self.tokenization_transform(fv_aho)["input_ids"]

        grafted_fv_aho_toks = self.graft_fv_germline_sequences(fv_aho_tok)

        return (
            self.process_grafted_seq_toks(grafted_fv_aho_toks),
            grafted_fv_aho_toks,
        )
