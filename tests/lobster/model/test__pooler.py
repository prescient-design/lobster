import numpy as np
import pytest
import torch
from lobster.model._mlp import LobsterMLP


@pytest.fixture
def amino_acid_input():
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    # sequences = list(torch.randint(0, 21, (4, 512, 72)))
    sequences = [["".join(list(np.random.choice(amino_acids, np.random.randint(520)))) for _ in range(4)]]
    ys = torch.randn([4, 1])
    return sequences, ys


@pytest.fixture
def plm_mean(amino_acid_input):
    model = LobsterMLP(model_name="MLM_mini", output_hidden=True, pooler="mean")
    model.eval()
    _, _, mean_hidden_states = model.predict_step(amino_acid_input, batch_idx=0)
    return model, mean_hidden_states


@pytest.fixture
def plm_cls(amino_acid_input):
    model = LobsterMLP(model_name="MLM_mini", output_hidden=True, pooler="cls")
    model.eval()
    _, _, cls_hidden_states = model.predict_step(amino_acid_input, batch_idx=0)
    return model, cls_hidden_states


@pytest.fixture
def plm_attn(amino_acid_input):
    model = LobsterMLP(model_name="MLM_mini", output_hidden=True, pooler="attn")
    model.eval()
    _, _, attn_hidden_states = model.predict_step(amino_acid_input, batch_idx=0)
    return model, attn_hidden_states


@pytest.fixture
def plm_weighted_mean(amino_acid_input):
    model = LobsterMLP(
        model_type="LobsterPCLM",
        model_name="CLM_mini",
        output_hidden=True,
        pooler="weighted_mean",
    )
    model.eval()
    _, _, weighted_mean_hidden_states = model.predict_step(amino_acid_input, batch_idx=0)
    return model, weighted_mean_hidden_states


# @pytest.fixture
# def cross_attention_pooler(max_length, scope="session"):
#     return CrossAttentionPooler()


class TestBasePoolers:
    def test_default_forward(self, plm_mean):
        model, hidden_states = plm_mean
        assert model.pooler_name == "mean"
        assert hidden_states.shape == torch.Size([4, 72])

    def test_cls_pooler(self, plm_cls, plm_mean):
        model, hidden_states = plm_cls
        _, mean_hidden_states = plm_mean
        assert model.pooler_name == "cls"
        assert hidden_states.shape == torch.Size([4, 72])

        assert (hidden_states != mean_hidden_states).all()

    def test_attn_pooler(self, plm_mean, plm_cls, plm_attn):
        model, hidden_states = plm_attn
        _, mean_hidden_states = plm_mean
        _, cls_hidden_states = plm_cls
        assert model.pooler_name == "attn"

        assert hidden_states.shape == torch.Size([4, 72])

        assert (hidden_states != mean_hidden_states).all() & (hidden_states != cls_hidden_states).all()

    def test_weighted_mean_pooler(self, plm_weighted_mean):
        model, hidden_states = plm_weighted_mean
        assert model.pooler_name == "weighted_mean"
        assert hidden_states.shape == torch.Size([4, 32])


# class TestCrossAttentionPooler:
#     def test_forward(self):
#         model = CrossAttentionPooler(config={})
#         model.eval()

#         hidden_states = torch.randn([4, 751, 1280])
#         amino_acid_input_mask = torch.randn([4, 751])

#         output = model(hidden_states, amino_acid_input_mask)

#         assert output.shape == torch.Size([4, 1280])
