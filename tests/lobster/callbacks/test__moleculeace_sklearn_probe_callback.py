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
            "mse": 2.37913,
            "r2": -0.01650,
            "spearman": 0.06732,
            "pearson": 0.05597,
        }

        assert "CHEMBL204_Ki" in results
        assert "mean" in results

        for metric, expected_value in expected_metrics.items():
            assert metric in results["CHEMBL204_Ki"]
            assert math.isclose(results["CHEMBL204_Ki"][metric], expected_value, rel_tol=1e-3)

            mean_value = results["mean"][metric]
            assert math.isclose(mean_value, expected_value, rel_tol=1e-3)

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
            "mse": 2.434589,
            "r2": -0.040197,
            "spearman": 0.000747,
            "pearson": 0.013382,
        }

        expected_chembl214 = {
            "mse": 1.332922,
            "r2": -0.030792,
            "spearman": 0.010840,
            "pearson": -0.004345,
        }

        expected_mean = {
            "mse": 1.883755,
            "r2": -0.035494,
            "spearman": 0.005793,
            "pearson": 0.004518,
        }

        # Check individual tasks
        for metric, expected_value in expected_chembl204.items():
            assert math.isclose(results["CHEMBL204_Ki"][metric], expected_value, rel_tol=1e-3), (
                f"Metric {metric} is not close to expected value {expected_value}. Got {results['CHEMBL204_Ki'][metric]}"
            )

        for metric, expected_value in expected_chembl214.items():
            assert math.isclose(results["CHEMBL214_Ki"][metric], expected_value, rel_tol=1e-3), (
                f"Metric {metric} is not close to expected value {expected_value}. Got {results['CHEMBL214_Ki'][metric]}"
            )

        # Check mean values
        for metric, expected_value in expected_mean.items():
            assert math.isclose(results["mean"][metric], expected_value, rel_tol=1e-3), (
                f"Metric {metric} is not close to expected value {expected_value}. Got {results['mean'][metric]}"
            )
