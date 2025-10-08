from unittest.mock import patch


from lobster.callbacks import PEERSklearnProbeCallback
from lobster.constants import PEERTask


class TestPEERSklearnProbeCallback:
    """Test suite for PEERSklearnProbeCallback functionality."""

    def test_initialization(self):
        """Test PEER callback initialization."""
        callback = PEERSklearnProbeCallback(seed=1)
        assert PEERTask.GB1 in callback.selected_tasks
        assert PEERTask.PROTEINNET not in callback.selected_tasks
        assert callback.probe_type == "linear"

        callback = PEERSklearnProbeCallback(tasks=[PEERTask.GB1, PEERTask.STABILITY], probe_type="elastic", seed=1)
        assert callback.selected_tasks == {PEERTask.GB1, PEERTask.STABILITY}
        assert callback.probe_type == "elastic"

    @patch("lobster.datasets.PEERDataset")
    def test_evaluate_single_task(self, mock_dataset_class, deterministic_model, mock_peer_dataset):
        """Test evaluation on single PEER task (GB1)."""

        def dataset_side_effect(task, split):
            return mock_peer_dataset(task, split)

        mock_dataset_class.side_effect = dataset_side_effect

        callback = PEERSklearnProbeCallback(tasks=[PEERTask.GB1], seed=1)

        results = callback.evaluate(deterministic_model)

        expected_metrics = {
            "mse": 1.727782964706421,
            "r2": -0.14102435111999512,
            "spearman": -0.00425225542858243,
            "pearson": -0.006624212488532066,
        }

        assert "gb1" in results
        assert "mean" in results

        for metric in expected_metrics.keys():
            assert metric in results["gb1"]
            assert isinstance(results["gb1"][metric], float)

            mean_value = results["mean"][metric]
            assert isinstance(mean_value, float)

            assert results["gb1"][metric] == expected_metrics[metric]
            assert results["mean"][metric] == expected_metrics[metric]

    @patch("lobster.datasets.PEERDataset")
    def test_evaluate_multiple_tasks(self, mock_dataset_class, deterministic_model, mock_peer_dataset):
        """Test evaluation on multiple PEER tasks."""

        def dataset_side_effect(task, split):
            return mock_peer_dataset(task, split)

        mock_dataset_class.side_effect = dataset_side_effect

        callback = PEERSklearnProbeCallback(tasks=[PEERTask.GB1, PEERTask.STABILITY], seed=1)

        results = callback.evaluate(deterministic_model)

        assert "gb1" in results
        assert "stability" in results
        assert "mean" in results

        expected_gb1 = {
            "mse": 1.727782964706421,
            "r2": -0.14102435111999512,
            "spearman": -0.00425225542858243,
            "pearson": -0.006624212488532066,
        }
        expected_stability = {
            "mse": 0.8437779545783997,
            "r2": -4.051199436187744,
            "spearman": 0.0027925781905651093,
            "pearson": 0.00850745104253292,
        }
        expected_mean = {
            "spearman": -0.0007298386190086603,
            "r2": -2.0961118936538696,
            "pearson": 0.0009416192770004272,
            "mse": 1.2857804596424103,
        }

        for metric in expected_gb1.keys():
            assert metric in results["gb1"]
            assert isinstance(results["gb1"][metric], float)
            assert results["gb1"][metric] == expected_gb1[metric]

        for metric in expected_stability.keys():
            assert metric in results["stability"]
            assert isinstance(results["stability"][metric], float)
            assert results["stability"][metric] == expected_stability[metric]

        for metric in expected_mean.keys():
            assert metric in results["mean"]
            assert isinstance(results["mean"][metric], float)
            assert results["mean"][metric] == expected_mean[metric]
