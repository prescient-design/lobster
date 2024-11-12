import importlib

import pandas as pd
import pytest
import torch
from lobster.tokenization import PmlmTokenizerTransform
from lobster.transforms.functional import GraftGermline


@pytest.fixture
def germline_df():
    return pd.DataFrame(
        {
            "fv_light_aho_tok": [
                torch.randint(0, 33, (512,)),
                torch.randint(0, 33, (512,)),
            ],
            "fv_heavy_aho_tok": [
                torch.randint(0, 33, (512,)),
                torch.randint(0, 33, (512,)),
            ],
            "fv_heavy_aho": ["A" * 298, "B" * 298],
            "fv_light_aho": ["A" * 298, "B" * 298],
            "HFR1": ["A" * 10, "B" * 10],
            "HFR2": ["A" * 10, "B" * 10],
            "HFR3a": ["A" * 10, "B" * 10],
            "HFR3b": ["A" * 10, "B" * 10],
            "HFR4": ["A" * 10, "B" * 10],
            "H1": ["A" * 10, "B" * 10],
            "H2": ["A" * 10, "B" * 10],
            "H3": ["A" * 10, "B" * 10],
            "H4": ["A" * 10, "B" * 10],
            "LFR1": ["A" * 10, "B" * 10],
            "LFR2": ["A" * 10, "B" * 10],
            "LFR3a": ["A" * 10, "B" * 10],
            "LFR3b": ["A" * 10, "B" * 10],
            "LFR4": ["A" * 10, "B" * 10],
            "L1": ["A" * 10, "B" * 10],
            "L2": ["A" * 10, "B" * 10],
            "L3": ["A" * 10, "B" * 10],
            "L4": ["A" * 10, "B" * 10],
            "heavy_v_gene": ["GENE1", "GENE2"],
            "heavy_j_gene": ["GENE1", "GENE2"],
            "light_v_gene": ["GENE1", "GENE2"],
            "light_j_gene": ["GENE1", "GENE2"],
        }
    )


@pytest.fixture
def tokenization_transform():
    path = importlib.resources.files("lobster") / "assets" / "pmlm_tokenizer"
    return PmlmTokenizerTransform(
        path,
        padding="max_length",
        truncation=True,
        max_length=512,
        mlm=True,
    )


class TestGraftGermline:
    def test__init__(self, germline_df: pd.DataFrame, tokenization_transform: PmlmTokenizerTransform):
        graft_germlines = GraftGermline(germline_df, tokenization_transform, chain="H", keep_const_idxs_list=[])
        assert "fv_heavy_aho" in graft_germlines.germline_df.columns
        assert "fv_light_aho" in graft_germlines.germline_df.columns
        assert graft_germlines.tokenization_transform is not None
        assert len(graft_germlines.idxs_to_graft) > 0

    def test__graft__(self, germline_df: pd.DataFrame, tokenization_transform: PmlmTokenizerTransform):
        graft_germlines = GraftGermline(germline_df, tokenization_transform, chain="H", keep_const_idxs_list=[])
        (
            processed_seq_df,
            grafted_fv_heavy_aho_toks,
        ) = graft_germlines.graft("C" * 298)
        assert isinstance(processed_seq_df, pd.DataFrame)
        assert isinstance(grafted_fv_heavy_aho_toks, torch.Tensor)

        assert len(set(processed_seq_df["fv_heavy_aho_germline"]) & set(processed_seq_df["fv_heavy_aho"])) == 0

    def test_vernier_constants(self, germline_df: pd.DataFrame, tokenization_transform: PmlmTokenizerTransform):
        graft_germlines_rabbit = GraftGermline(
            germline_df, tokenization_transform, keep_vernier_from_animal="rabbit", chain="H", keep_const_idxs_list=[]
        )
        processed_seq_df, grafted_fv_heavy_aho_toks_rabbit = graft_germlines_rabbit.graft("C" * 298)

        graft_germlines_other = GraftGermline(
            germline_df,
            tokenization_transform,
            keep_vernier_from_animal="rabbit",
            chain="H",
            keep_const_idxs_list=[1, 2, 3, 4],
        )
        processed_seq_df, grafted_fv_heavy_aho_toks_custom = graft_germlines_other.graft("C" * 298)

        graft_germlines_ref = GraftGermline(germline_df, tokenization_transform, chain="H", keep_const_idxs_list=[])
        processed_seq_df, grafted_fv_heavy_aho_toks_ref = graft_germlines_ref.graft("C" * 298)

        assert not torch.allclose(grafted_fv_heavy_aho_toks_rabbit, grafted_fv_heavy_aho_toks_custom)
        assert not torch.allclose(grafted_fv_heavy_aho_toks_rabbit, grafted_fv_heavy_aho_toks_ref)
