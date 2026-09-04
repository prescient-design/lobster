"""Tests for MoleculeNetSklearnProbeCallback."""

from unittest.mock import patch

import pytest

from lobster.callbacks import MoleculeNetSklearnProbeCallback
from lobster.constants import MOLECULENET_TASK_NAMES


class TestMoleculeNetSklearnProbeCallback:
    """Test suite for MoleculeNetSklearnProbeCallback functionality."""

    def test_initialization(self):
        """Test MoleculeNet callback initialization defaults and overrides."""
        callback = MoleculeNetSklearnProbeCallback(seed=1)
        assert callback.tasks == MOLECULENET_TASK_NAMES
        assert callback.probe_type == "linear"
        assert callback.test_size == 0.2
        assert callback.max_samples is None

        callback = MoleculeNetSklearnProbeCallback(
            tasks=["BBBP", "ESOL"],
            probe_type="elastic",
            test_size=0.3,
            max_samples=100,
            seed=1,
        )
        assert callback.tasks == {"BBBP", "ESOL"}
        assert callback.probe_type == "elastic"
        assert callback.test_size == 0.3
        assert callback.max_samples == 100

    def test_initialization_rejects_unknown_tasks(self):
        with pytest.raises(ValueError, match="Unknown MoleculeNet tasks"):
            MoleculeNetSklearnProbeCallback(tasks=["not_a_task"], seed=1)

    @patch("lobster.callbacks._moleculenet_sklearn_probe_callback.MoleculeNetDataset")
    def test_evaluate_binary_task(self, mock_dataset_class, deterministic_model, mock_moleculenet_dataset):
        """Test evaluation on a binary MoleculeNet task (BBBP)."""
        mock_dataset_class.return_value = mock_moleculenet_dataset("BBBP")

        callback = MoleculeNetSklearnProbeCallback(tasks=["BBBP"], seed=1, max_samples=10)
        results = callback.evaluate(deterministic_model)

        assert "BBBP" in results
        assert "mean" in results
        assert isinstance(results["BBBP"], dict)

        for metric in ("accuracy", "auroc", "f1"):
            assert metric in results["BBBP"]
            assert isinstance(results["BBBP"][metric], float)
            assert metric in results["mean"]

    @patch("lobster.callbacks._moleculenet_sklearn_probe_callback.MoleculeNetDataset")
    def test_evaluate_regression_task(self, mock_dataset_class, deterministic_model, mock_moleculenet_dataset):
        """Test evaluation on a regression MoleculeNet task (ESOL)."""
        mock_dataset_class.return_value = mock_moleculenet_dataset("ESOL")

        callback = MoleculeNetSklearnProbeCallback(tasks=["ESOL"], seed=1, max_samples=10)
        results = callback.evaluate(deterministic_model)

        assert "ESOL" in results
        assert "mean" in results
        assert isinstance(results["ESOL"], dict)

        for metric in ("mse", "r2"):
            assert metric in results["ESOL"]
            assert isinstance(results["ESOL"][metric], float)
            assert metric in results["mean"]
