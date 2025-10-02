"""Tests for MoleculeACESklearnProbeCallback."""

import math
from unittest.mock import patch

from lobster.callbacks import MoleculeACESklearnProbeCallback
from lobster.constants import MOLECULEACE_TASKS


class TestMoleculeACESklearnProbeCallback:
    """Test suite for MoleculeACESklearnProbeCallback functionality."""

    def test_initialization(self):
        """Test MoleculeACE callback initialization."""
        callback = MoleculeACESklearnProbeCallback(seed=1)
        assert callback.tasks == MOLECULEACE_TASKS
        assert callback.use_protein_sequences is False

        callback = MoleculeACESklearnProbeCallback(
            tasks=["CHEMBL204_Ki", "CHEMBL214_Ki"], use_protein_sequences=True, seed=1
        )
        assert callback.tasks == {"CHEMBL204_Ki", "CHEMBL214_Ki"}
        assert callback.use_protein_sequences is True

    @patch("lobster.datasets.MoleculeACEDataset")
    def test_evaluate_single_task(self, mock_dataset_class, deterministic_model, mock_moleculeace_dataset):
        """Test evaluation on single MoleculeACE task."""

        # Setup mock datasets for train/test
        def dataset_side_effect(task, train, include_protein_sequences=False):
            return mock_moleculeace_dataset(task, train, include_protein_sequences)

        mock_dataset_class.side_effect = dataset_side_effect

        callback = MoleculeACESklearnProbeCallback(tasks=["CHEMBL204_Ki"], seed=1)

        results = callback.evaluate(deterministic_model)

        # Verify structure and expected values
        expected_metrics = {
            "mse": 2.3791260719299316,
            "r2": -0.016499887165573046,
            "spearman": 0.06732063740491867,
            "pearson": 0.0559663400053978,
        }

        assert "CHEMBL204_Ki" in results
        assert "mean" in results

        for metric, expected_value in expected_metrics.items():
            assert metric in results["CHEMBL204_Ki"]
            assert math.isclose(results["CHEMBL204_Ki"][metric], expected_value, rel_tol=1e-4)

            mean_value = results["mean"][metric]
            assert math.isclose(mean_value, expected_value, rel_tol=1e-4)

    @patch("lobster.datasets.MoleculeACEDataset")
    def test_evaluate_multiple_tasks(self, mock_dataset_class, deterministic_model, mock_moleculeace_dataset):
        """Test evaluation on multiple tasks."""

        def dataset_side_effect(task, train, include_protein_sequences=False):
            return mock_moleculeace_dataset(task, train, include_protein_sequences)

        mock_dataset_class.side_effect = dataset_side_effect

        callback = MoleculeACESklearnProbeCallback(tasks=["CHEMBL204_Ki", "CHEMBL214_Ki"], seed=1)

        results = callback.evaluate(deterministic_model)

        # Should have results for both tasks plus mean
        assert "CHEMBL204_Ki" in results
        assert "CHEMBL214_Ki" in results
        assert "mean" in results

        # Verify individual task results
        expected_chembl204 = {
            "mse": 2.3791260719299316,
            "r2": -0.016499887165573046,
            "spearman": 0.06732063740491867,
            "pearson": 0.0559663400053978,
        }

        expected_chembl214 = {
            "mse": 1.320008397102356,
            "r2": -0.020805881232939916,
            "spearman": 0.001118487911298871,
            "pearson": 0.0036589554511010647,
        }

        expected_mean = {
            "mse": 1.8495672345161438,
            "r2": -0.01865288419925648,
            "spearman": 0.03421956265810877,
            "pearson": 0.02981264772824943,
        }

        # Check individual tasks
        for metric, expected_value in expected_chembl204.items():
            assert math.isclose(results["CHEMBL204_Ki"][metric], expected_value, rel_tol=1e-4)

        for metric, expected_value in expected_chembl214.items():
            assert math.isclose(results["CHEMBL214_Ki"][metric], expected_value, rel_tol=1e-4)

        # Check mean values
        for metric, expected_value in expected_mean.items():
            assert math.isclose(results["mean"][metric], expected_value, rel_tol=1e-4)
