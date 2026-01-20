"""Tests for SklearnProbeCallback base class."""

import math
import pytest
import torch
from torch.utils.data import Subset

from lobster.callbacks import SklearnProbeCallback, SklearnProbeTaskConfig


class TestSklearnProbeCallback:
    """Test suite for SklearnProbeCallback base functionality."""

    @pytest.fixture
    def callback(self):
        return SklearnProbeCallback(seed=1)

    def test_get_embeddings(self, callback, deterministic_model, mock_regression_dataset):
        """Test embedding extraction produces expected shapes."""

        embeddings, targets = callback.get_embeddings(deterministic_model, mock_regression_dataset)

        # Test shapes
        assert embeddings.shape == (10, 64)  # 10 samples, 64 dims
        assert targets.shape == (10,)  # 10 targets

        # Test that embeddings are tensors
        assert isinstance(embeddings, torch.Tensor)
        assert isinstance(targets, torch.Tensor)

    def test_train_and_evaluate_probe_regression(self, callback, deterministic_model, mock_regression_dataset):
        """Test probe training and evaluation for regression."""

        config = SklearnProbeTaskConfig(task_name="test_regression", task_type="regression", probe_type="linear")

        train_dataset = Subset(mock_regression_dataset, [0, 1, 2, 3, 4])
        test_dataset = Subset(mock_regression_dataset, [5, 6, 7, 8])

        result = callback.train_and_evaluate_probe_on_task(deterministic_model, train_dataset, test_dataset, config)

        expected_metrics = {
            "mse": 1.124,
            "r2": 0.01560,
            "spearman": 0.4000,
            "pearson": 0.1544,
        }

        assert isinstance(result.metrics, dict)
        for metric, expected_value in expected_metrics.items():
            assert metric in result.metrics
            assert math.isclose(result.metrics[metric], expected_value, rel_tol=1e-3), (
                f"Metric {metric} is not close to expected value {expected_value}. Got {result.metrics[metric]}"
            )

    def test_train_and_evaluate_probe_binary_classification(
        self, callback, deterministic_model, mock_binary_classification_dataset
    ):
        """Test probe training and evaluation for binary classification."""

        config = SklearnProbeTaskConfig(
            task_name="test_classification", task_type="binary", probe_type="linear", num_classes=2
        )

        train_dataset = Subset(mock_binary_classification_dataset, [0, 1])
        test_dataset = Subset(mock_binary_classification_dataset, [2, 3])

        result = callback.train_and_evaluate_probe_on_task(deterministic_model, train_dataset, test_dataset, config)

        expected_metrics = {"accuracy": 1.0, "f1": 1.0, "f1_weighted": 1.0, "auroc": 1.0}

        assert isinstance(result.metrics, dict)

        for metric, expected_value in expected_metrics.items():
            assert metric in result.metrics
            assert math.isclose(result.metrics[metric], expected_value, rel_tol=1e-4), (
                f"Metric {metric} is not close to expected value {expected_value}. Got {result.metrics[metric]}"
            )

    def test_train_and_evaluate_probe_multilabel_classification(
        self, callback, deterministic_model, mock_multilabel_dataset
    ):
        """Test probe training and evaluation for multilabel classification."""

        config = SklearnProbeTaskConfig(
            task_name="test_multilabel",
            task_type="multilabel",
            probe_type="linear",
            num_classes=5,
            classification_threshold=0.5,
        )

        train_dataset = Subset(mock_multilabel_dataset, [0, 1, 2])
        test_dataset = Subset(mock_multilabel_dataset, [0, 1, 2])

        result = callback.train_and_evaluate_probe_on_task(deterministic_model, train_dataset, test_dataset, config)

        expected_metrics = {
            "accuracy": 0.6000,
            "f1": 0.2500,
            "f1_weighted": 0.1000,
            "auroc": 0.2000,
        }

        assert isinstance(result.metrics, dict)
        for metric, expected_value in expected_metrics.items():
            assert metric in result.metrics
            assert math.isclose(result.metrics[metric], expected_value, rel_tol=1e-3), (
                f"Metric {metric} is not close to expected value {expected_value}. Got {result.metrics[metric]}"
            )

    def test_cross_validation(self, callback, deterministic_model, mock_regression_dataset):
        """Test cross-validation functionality."""

        config = SklearnProbeTaskConfig(task_name="test_cv", task_type="regression", probe_type="linear")

        result = callback.train_and_evaluate_cv_probe_on_task(
            deterministic_model, mock_regression_dataset, config, n_folds=3
        )

        expected_metrics = {
            "mse": 1.231,
            "r2": -1.589,
            "spearman": -0.3000,
            "pearson": -0.2828,
        }

        assert isinstance(result.metrics, dict)
        for metric, expected_value in expected_metrics.items():
            assert metric in result.metrics
            assert math.isclose(result.metrics[metric], expected_value, rel_tol=1e-3), (
                f"Metric {metric} is not close to expected value {expected_value}"
            )
