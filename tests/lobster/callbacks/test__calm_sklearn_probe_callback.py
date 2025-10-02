"""Tests for CalmSklearnProbeCallback."""

import math
from unittest.mock import patch

from lobster.callbacks import CalmSklearnProbeCallback


class TestCalmSklearnProbeCallback:
    """Test suite for CalmSklearnProbeCallback functionality."""

    def test_initialization(self):
        """Test CALM callback initialization."""
        callback = CalmSklearnProbeCallback(seed=1)
        assert "meltome" in callback.tasks
        assert "solubility" in callback.tasks
        assert callback.species == {"hsapiens", "ecoli", "scerevisiae"}

        callback = CalmSklearnProbeCallback(tasks=["meltome", "solubility"], species=["hsapiens"], seed=1)
        assert callback.tasks == {"meltome", "solubility"}
        assert callback.species == {"hsapiens"}

    @patch("lobster.datasets.CalmPropertyDataset")
    def test_evaluate_regression_task(self, mock_dataset_class, deterministic_model, mock_calm_dataset):
        """Test evaluation on regression task (meltome)."""
        mock_dataset_class.return_value = mock_calm_dataset("meltome")

        callback = CalmSklearnProbeCallback(tasks=["meltome"], species=None, seed=1, max_samples=10)

        results = callback.evaluate(deterministic_model)

        expected_metrics = {
            "mse": 0.9771747589111328,
            "r2": 0.9501152038574219,
            "spearman": 0.9999959468841553,
            "pearson": 1.0,
        }

        assert "meltome" in results
        assert "mean" in results
        assert isinstance(results["meltome"], dict)

        for metric, expected_value in expected_metrics.items():
            assert metric in results["meltome"]
            assert math.isclose(results["meltome"][metric], expected_value, rel_tol=1e-4)

            mean_value = results["mean"][metric]
            assert math.isclose(mean_value, expected_value, rel_tol=1e-4)

    @patch("lobster.datasets.CalmPropertyDataset")
    def test_evaluate_cross_validation_mode(self, mock_dataset_class, deterministic_model, mock_calm_dataset):
        """Test cross-validation evaluation mode."""
        mock_dataset_class.return_value = mock_calm_dataset("meltome")

        callback = CalmSklearnProbeCallback(
            tasks=["meltome"], use_cross_validation=True, n_folds=3, seed=1, max_samples=10
        )

        results = callback.evaluate(deterministic_model)

        expected_metrics = {
            "mse": 74.91061846415202,
            "r2": -11.522627671559652,
            "spearman": -0.9333321452140808,
            "pearson": -0.8949546019236246,
        }

        assert "meltome" in results
        for metric, expected_value in expected_metrics.items():
            assert metric in results["meltome"]
            assert math.isclose(results["meltome"][metric], expected_value, rel_tol=1e-4)

            mean_value = results["mean"][metric]
            assert math.isclose(mean_value, expected_value, rel_tol=1e-4)
