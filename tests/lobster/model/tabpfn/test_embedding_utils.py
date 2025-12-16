import pytest
import torch
import lightning.pytorch as pl

from lobster.model.tabpfn._embedding_utils import (
    initialize_embedding_model,
    extract_embeddings,
    get_embedding_dim,
    _pool_embeddings,
    _concatenate_additional_features,
)


class MockEmbeddingModel(pl.LightningModule):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def __call__(self, input_ids, attention_mask, output_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        last_hidden = torch.randn(batch_size, seq_len, hidden_size)

        return {
            "hidden_states": [last_hidden, last_hidden],
            "last_hidden_state": last_hidden,
        }


class MockEmbeddingModelWithModel(pl.LightningModule):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.model = MockEmbeddingModel(hidden_size)


def test_initialize_embedding_model_validation():
    with pytest.raises(ValueError, match="Must provide either"):
        initialize_embedding_model("LobsterPMLM", None, None, 512)

    with pytest.raises(ValueError, match="Must provide either"):
        initialize_embedding_model("LobsterPMLM", "some_model", "some_checkpoint", 512)


def test_get_embedding_dim():
    model = MockEmbeddingModel(hidden_size=256)
    dim = get_embedding_dim(model, num_chains=1)
    assert dim == 256

    dim = get_embedding_dim(model, num_chains=3)
    assert dim == 768


def test_get_embedding_dim_with_embedding_dim_attr():
    model = pl.LightningModule()
    model.embedding_dim = 512
    dim = get_embedding_dim(model, num_chains=2)
    assert dim == 1024


def test_get_embedding_dim_raises_error():
    model = pl.LightningModule()
    with pytest.raises(ValueError, match="Cannot determine hidden size"):
        get_embedding_dim(model, num_chains=1)


def test_pool_embeddings_mean():
    batch_size, seq_len, hidden = 2, 5, 8
    hidden_states = torch.randn(batch_size, seq_len, hidden)
    attention_mask = torch.ones(batch_size, seq_len)

    pooled = _pool_embeddings(hidden_states, attention_mask, "mean")
    assert pooled.shape == (batch_size, hidden)


def test_pool_embeddings_max():
    batch_size, seq_len, hidden = 2, 5, 8
    hidden_states = torch.randn(batch_size, seq_len, hidden)
    attention_mask = torch.ones(batch_size, seq_len)

    pooled = _pool_embeddings(hidden_states, attention_mask, "max")
    assert pooled.shape == (batch_size, hidden)


def test_pool_embeddings_cls():
    batch_size, seq_len, hidden = 2, 5, 8
    hidden_states = torch.randn(batch_size, seq_len, hidden)
    attention_mask = torch.ones(batch_size, seq_len)

    pooled = _pool_embeddings(hidden_states, attention_mask, "cls")
    assert pooled.shape == (batch_size, hidden)
    assert torch.allclose(pooled, hidden_states[:, 0, :])


def test_pool_embeddings_invalid():
    hidden_states = torch.randn(2, 5, 8)
    attention_mask = torch.ones(2, 5)

    with pytest.raises(ValueError, match="Unknown pooling method"):
        _pool_embeddings(hidden_states, attention_mask, "invalid")


def test_concatenate_additional_features():
    batch_size, hidden = 3, 8
    embeddings = torch.randn(batch_size, hidden)

    batch = {
        "feat1": torch.randn(batch_size, 2),
        "feat2": torch.randn(batch_size),
    }

    result = _concatenate_additional_features(embeddings, batch, ["feat1", "feat2"])
    assert result.shape == (batch_size, hidden + 2 + 1)


def test_concatenate_additional_features_no_features():
    batch_size, hidden = 3, 8
    embeddings = torch.randn(batch_size, hidden)
    batch = {}

    result = _concatenate_additional_features(embeddings, batch, [])
    assert result.shape == embeddings.shape
    assert torch.allclose(result, embeddings)


def test_concatenate_additional_features_missing_features():
    batch_size, hidden = 3, 8
    embeddings = torch.randn(batch_size, hidden)

    batch = {"feat1": torch.randn(batch_size, 2)}

    result = _concatenate_additional_features(embeddings, batch, ["feat1", "missing"])
    assert result.shape == (batch_size, hidden + 2)


def test_extract_embeddings_with_model():
    model = MockEmbeddingModelWithModel(hidden_size=64)
    batch_size, seq_len = 4, 10

    batch = {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }

    embeddings = extract_embeddings(model, batch, "mean", freeze_embeddings=True, additional_features=[])
    assert embeddings.shape == (batch_size, 64)


def test_extract_embeddings_without_model():
    model = MockEmbeddingModel(hidden_size=64)
    batch_size, seq_len = 4, 10

    batch = {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
    }

    embeddings = extract_embeddings(model, batch, "cls", freeze_embeddings=False, additional_features=[])
    assert embeddings.shape == (batch_size, 64)


def test_extract_embeddings_with_additional_features():
    model = MockEmbeddingModel(hidden_size=32)
    batch_size, seq_len = 2, 8

    batch = {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "extra_feat": torch.randn(batch_size, 5),
    }

    embeddings = extract_embeddings(model, batch, "max", freeze_embeddings=True, additional_features=["extra_feat"])
    assert embeddings.shape == (batch_size, 32 + 5)
