import math
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
    def test_evaluate(self, mock_dataset_class, deterministic_model, mock_peer_dataset):
        """Test evaluation on multiple PEER tasks - both should return only spearman metric."""

        def dataset_side_effect(task, split):
            return mock_peer_dataset(task, split)

        mock_dataset_class.side_effect = dataset_side_effect

        callback = PEERSklearnProbeCallback(tasks=[PEERTask.GB1, PEERTask.STABILITY], seed=1)

        results = callback.evaluate(deterministic_model)

        assert "gb1" in results
        assert "stability" in results
        assert "mean" in results

        # Both GB1 and STABILITY are in PEER_TASK_METRICS and should only return spearman
        expected_gb1_spearman = -0.004252
        expected_stability_spearman = 0.002793
        expected_mean_spearman = -0.0007298

        assert len(results["gb1"]) == 1
        assert math.isclose(results["gb1"]["spearman"], expected_gb1_spearman, rel_tol=1e-3)

        assert len(results["stability"]) == 1
        assert math.isclose(results["stability"]["spearman"], expected_stability_spearman, rel_tol=1e-3)

        assert len(results["mean"]) == 1
        assert math.isclose(results["mean"]["spearman"], expected_mean_spearman, rel_tol=1e-3)
