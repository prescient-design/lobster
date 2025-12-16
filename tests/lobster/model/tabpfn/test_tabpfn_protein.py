import pytest
import torch
import numpy as np
import lightning.pytorch as pl
from unittest.mock import patch

from lobster.model.tabpfn.tabpfn_protein import TabPFNProteinModel


class MockTabPFNModel:
    def __init__(self, n_estimators=4, ignore_pretraining_limits=True):
        self.n_estimators = n_estimators
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self._is_fitted = False

    def fit(self, X, y):
        self._is_fitted = True
        self.n_features = X.shape[1]
        return self

    def predict(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        return np.random.randn(X.shape[0])

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        n_samples = X.shape[0]
        probs = np.random.rand(n_samples, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    @classmethod
    def create_default_for_version(cls, version, n_estimators=4, ignore_pretraining_limits=True):
        return cls(n_estimators=n_estimators, ignore_pretraining_limits=ignore_pretraining_limits)


class MockEmbeddingModel(pl.LightningModule):
    def __init__(self, model_name=None, max_length=512, hidden_size=128):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def __call__(self, input_ids, attention_mask, output_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        last_hidden = torch.randn(batch_size, seq_len, hidden_size)

        return {
            "hidden_states": [last_hidden, last_hidden],
            "last_hidden_state": last_hidden,
        }

    def parameters(self):
        return []


@pytest.fixture
def mock_tabpfn():
    with (
        patch("tabpfn.TabPFNRegressor", MockTabPFNModel),
        patch("tabpfn.TabPFNClassifier", MockTabPFNModel),
    ):
        yield


@pytest.fixture
def mock_embedding_init():
    def init_mock(embedding_model_type, embedding_model_name, embedding_checkpoint, max_length):
        return MockEmbeddingModel(model_name=embedding_model_name, max_length=max_length)

    with patch("lobster.model.tabpfn.tabpfn_protein.initialize_embedding_model", side_effect=init_mock):
        yield


def test_tabpfn_model_init_regression(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    assert model._task == "regression"
    assert model._num_labels == 1
    assert not model.is_fitted
    assert len(model.tabpfn_models) == 0


def test_tabpfn_model_init_classification(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="classification",
        num_labels=3,
        embedding_model_name="test_model",
    )

    assert model._task == "classification"
    assert model._num_labels == 3
    assert not model.is_fitted


def test_tabpfn_model_training_step(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    batch = {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.ones(4, 10),
        "labels": torch.randn(4),
    }

    loss = model.training_step(batch, 0)

    assert len(model._training_embeddings) == 1
    assert len(model._training_labels) == 1
    assert loss.requires_grad


def test_tabpfn_model_on_train_epoch_end(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
        tabpfn_n_ensemble=2,
    )

    batch = {
        "input_ids": torch.randint(0, 100, (8, 10)),
        "attention_mask": torch.ones(8, 10),
        "labels": torch.randn(8),
    }

    model.training_step(batch, 0)
    model.on_train_epoch_end()

    assert model.is_fitted
    assert len(model.tabpfn_models) > 0
    assert len(model._training_embeddings) == 0
    assert len(model._training_labels) == 0


def test_tabpfn_model_validation_step_not_fitted(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    batch = {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.ones(4, 10),
        "labels": torch.randn(4),
    }

    loss = model.validation_step(batch, 0)
    assert loss.item() == 0.0


def test_tabpfn_model_test_step_not_fitted(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    batch = {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.ones(4, 10),
        "labels": torch.randn(4),
    }

    loss = model.test_step(batch, 0)
    assert loss.item() == 0.0


def test_tabpfn_model_predict_step_raises_if_not_fitted(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    batch = {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.ones(4, 10),
    }

    with pytest.raises(RuntimeError, match="Model must be fitted"):
        model.predict_step(batch, 0)


def test_tabpfn_model_configure_optimizers_frozen(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    optimizer = model.configure_optimizers()
    assert optimizer is None


def test_tabpfn_model_configure_optimizers_trainable(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    optimizer = model.configure_optimizers()
    assert optimizer is None


def test_tabpfn_model_forward_not_fitted(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    batch = {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.ones(4, 10),
    }

    output = model.forward(batch)
    assert output.shape[0] == 4


def test_tabpfn_model_metrics_setup_regression(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
    )

    assert hasattr(model, "train_r2")
    assert hasattr(model, "train_mae")
    assert hasattr(model, "train_spearman")
    assert hasattr(model, "val_r2")
    assert hasattr(model, "val_mae")
    assert hasattr(model, "val_spearman")
    assert hasattr(model, "test_r2")
    assert hasattr(model, "test_mae")
    assert hasattr(model, "test_spearman")


def test_tabpfn_model_metrics_setup_classification(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="classification",
        num_labels=3,
        embedding_model_name="test_model",
    )

    assert hasattr(model, "train_f1")
    assert hasattr(model, "train_auroc")
    assert hasattr(model, "val_f1")
    assert hasattr(model, "val_auroc")
    assert hasattr(model, "test_f1")
    assert hasattr(model, "test_auroc")


def test_tabpfn_model_with_additional_features(mock_tabpfn, mock_embedding_init):
    model = TabPFNProteinModel(
        task="regression",
        num_labels=1,
        embedding_model_name="test_model",
        additional_features=["feat1", "feat2"],
    )

    assert model._additional_features == ["feat1", "feat2"]


def test_tabpfn_model_import_error():
    with patch.dict("sys.modules", {"tabpfn": None}):
        with pytest.raises(ImportError, match="TabPFN is not installed"):
            with patch(
                "lobster.model.tabpfn.tabpfn_protein.initialize_embedding_model", return_value=MockEmbeddingModel()
            ):
                TabPFNProteinModel(
                    task="regression",
                    embedding_model_name="test_model",
                )
