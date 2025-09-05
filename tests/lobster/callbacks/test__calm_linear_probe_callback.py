from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

# Direct imports to avoid rdkit dependency chain
from lobster.callbacks._calm_linear_probe_callback import CalmLinearProbeCallback
from lobster.constants._calm_tasks import CALM_TASKS

# Add a constant for the max_length parameter used in tests
MAX_LENGTH = 1024


class MockDataset(Dataset):
    def __init__(self, size=10, task_type="regression"):
        self.size = size
        self.task_type = task_type

        # Generate random data
        if task_type == "regression":
            self.targets = torch.randn(size)
        elif task_type == "binary":
            self.targets = torch.randint(0, 2, (size,), dtype=torch.long)
        elif task_type == "multiclass":
            self.num_classes = 10
            self.targets = torch.randint(0, self.num_classes, (size,), dtype=torch.long)
        elif task_type == "multilabel":
            self.num_classes = 10  # Match localization task
            self.targets = torch.randint(0, 2, (size, self.num_classes), dtype=torch.long)
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
        # Generate shorter random sequences
        self.sequences = []
        for _ in range(size):
            sequence = "".join(np.random.choice(["A", "C", "G", "T"]) for _ in range(10))
            self.sequences.append({"text": sequence})

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class MockModelWithEmbeddings(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size

    def tokens_to_latents(self, input_ids, **kwargs):
        batch_size = input_ids.size(0)
        return torch.randn(batch_size, MAX_LENGTH, self.hidden_size)


class MockLightningModule(torch.nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = "cpu"
        self.model = MockModelWithEmbeddings(hidden_size)

    def embed_sequences(self, sequences, modality="nucleotide", aggregate=True):
        """Mock embed_sequences method that mimics UME's behavior."""
        batch_size = len(sequences)
        if aggregate:
            # Return aggregated embeddings (batch_size, hidden_size)
            return torch.randn(batch_size, self.hidden_size)
        else:
            # Return per-token embeddings (batch_size, seq_len, hidden_size)
            seq_len = MAX_LENGTH  # Use the same max length as tokenizer
            return torch.randn(batch_size, seq_len, self.hidden_size)


@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.current_epoch = 0
    trainer.logger = Mock()
    trainer.global_rank = 0  # Add this to prevent skipping validation
    return trainer


@pytest.fixture
def mock_pl_module():
    return MockLightningModule()


def test_callback_initialization():
    """Test basic initialization of the callback."""
    callback = CalmLinearProbeCallback(
        max_length=MAX_LENGTH, tasks=["meltome", "solubility"], species=["hsapiens", "ecoli"], batch_size=16
    )

    assert callback.tasks == {"meltome", "solubility"}
    assert callback.species == {"hsapiens", "ecoli"}
    assert callback.batch_size == 16
    assert callback.test_size == 0.2
    assert callback.max_samples == 3000


@pytest.mark.parametrize(
    "task",
    [
        "meltome",  # regression task
        "localization",  # multiclass task
    ],
)
def test_single_task_evaluation(task, mock_trainer, mock_pl_module):
    """Test evaluation of individual tasks."""
    task_type, num_classes = CALM_TASKS[task]

    callback = CalmLinearProbeCallback(max_length=MAX_LENGTH, tasks=[task])

    mock_dataset = MockDataset(size=10, task_type=task_type)

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        callback.task_type = task_type  # Need to explicitly set the task type before evaluation

        if num_classes:
            callback.num_classes = num_classes

        mock_pl_module.device = "cpu"
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Verify metrics were logged
    mock_trainer.logger.log_metrics.assert_called()


@pytest.mark.parametrize(
    "task,species",
    [
        ("protein_abundance", "hsapiens"),
        ("transcript_abundance", "ecoli"),
    ],
)
def test_species_specific_task_evaluation(task, species, mock_trainer, mock_pl_module):
    """Test evaluation of species-specific tasks."""
    task_type, num_classes = CALM_TASKS[task]

    callback = CalmLinearProbeCallback(max_length=MAX_LENGTH, tasks=[task], species=[species])

    mock_dataset = MockDataset(size=50, task_type=task_type)

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        callback.task_type = task_type
        if num_classes:
            callback.num_classes = num_classes

        mock_pl_module.device = "cpu"
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    # Verify metrics were logged for the specific species
    calls = mock_trainer.logger.log_metrics.call_args_list
    logged_keys = set()
    for call in calls:
        args, _ = call
        logged_keys.update(args[0].keys())

    expected_task_key = f"{task}_{species}"
    assert any(expected_task_key in key for key in logged_keys)


def test_dataset_caching():
    """Test that dataset splits are properly cached."""
    callback = CalmLinearProbeCallback(max_length=MAX_LENGTH, tasks=["meltome"])

    mock_dataset = MockDataset(size=50)

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        # First call should create and cache the split
        train1, test1 = callback._create_split_datasets("meltome")
        # Second call should return cached split
        train2, test2 = callback._create_split_datasets("meltome")

        # Verify same splits are returned
        assert id(train1) == id(train2)
        assert id(test1) == id(test2)


def test_max_samples_limit():
    """Test that datasets are properly subsampled when exceeding max_samples."""
    max_samples = 100
    callback = CalmLinearProbeCallback(max_length=MAX_LENGTH, tasks=["meltome"], max_samples=max_samples)

    mock_dataset = MockDataset(size=max_samples * 2)

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        train, test = callback._create_split_datasets("meltome")

        assert len(train) + len(test) <= max_samples


def test_aggregate_metrics_reset(mock_trainer, mock_pl_module):
    """Test that aggregate metrics are properly reset between epochs."""
    callback = CalmLinearProbeCallback(max_length=MAX_LENGTH, tasks=["meltome"])
    callback.task_type = "regression"  # Set task type explicitly

    mock_dataset = MockDataset(size=10, task_type="regression")

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        # First epoch
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
        first_metrics = callback.aggregate_metrics.copy()

        # Second epoch
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
        second_metrics = callback.aggregate_metrics

        # Verify metrics were reset
        assert first_metrics.keys() == second_metrics.keys()
        assert all(len(first_metrics[k]) == len(second_metrics[k]) for k in first_metrics)


@pytest.mark.parametrize("use_cross_validation", [True, False])
@pytest.mark.parametrize("task_type", ["regression", "multilabel"])
def test_cross_validation_functionality(use_cross_validation, task_type, mock_trainer, mock_pl_module):
    """Test cross-validation vs train/test split for different task types."""
    # Choose appropriate task based on task_type
    if task_type == "regression":
        task = "meltome"
        num_classes = None
    else:  # multilabel
        task = "localization"
        num_classes = 10

    callback = CalmLinearProbeCallback(
        max_length=MAX_LENGTH,
        tasks=[task],
        use_cross_validation=use_cross_validation,
        n_folds=3,  # Use fewer folds for faster testing
        batch_size=8,
    )

    mock_dataset = MockDataset(size=30, task_type=task_type)  # Larger dataset for CV

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        # Set task type and num_classes
        callback.task_type = task_type
        if num_classes:
            callback.num_classes = num_classes

        mock_pl_module.device = "cpu"
        results = callback.evaluate(mock_pl_module, mock_trainer)

        # Verify results structure
        assert isinstance(results, dict)
        assert task in results or "mean" in results

        # For cross-validation, check that we get std metrics
        if use_cross_validation and task in results:
            task_metrics = results[task]
            if task_type == "regression":
                assert "mse_std" in task_metrics or "r2_std" in task_metrics
            else:  # multilabel
                assert "accuracy_std" in task_metrics or "f1_std" in task_metrics


@pytest.mark.parametrize("dimensionality_reduction", [True, False])
def test_dimensionality_reduction(dimensionality_reduction, mock_trainer, mock_pl_module):
    """Test PCA dimensionality reduction functionality."""
    callback = CalmLinearProbeCallback(
        max_length=MAX_LENGTH,
        tasks=["meltome"],
        dimensionality_reduction=dimensionality_reduction,
        reduced_dim=16,
        batch_size=8,
    )

    mock_dataset = MockDataset(size=20, task_type="regression")

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        callback.task_type = "regression"
        mock_pl_module.device = "cpu"

        results = callback.evaluate(mock_pl_module, mock_trainer)

        # Verify results are generated
        assert isinstance(results, dict)

        # If dimensionality reduction is used, check that dim_reducers are created
        if dimensionality_reduction:
            assert len(callback.dim_reducers) > 0
            # Check that PCA was fitted
            for task_key, pca in callback.dim_reducers.items():
                assert hasattr(pca, "components_")
                assert pca.n_components_ > 0
        else:
            assert len(callback.dim_reducers) == 0


@pytest.mark.parametrize("probe_type", ["linear", "elastic", "svm"])
@pytest.mark.parametrize("task_type", ["regression", "multilabel"])
def test_different_probe_types(probe_type, task_type, mock_trainer, mock_pl_module):
    """Test different probe types (linear, elastic, svm) for various task types."""
    # Choose appropriate task based on task_type
    if task_type == "regression":
        task = "meltome"
        num_classes = None
    else:  # multilabel
        task = "localization"
        num_classes = 10

    callback = CalmLinearProbeCallback(
        max_length=MAX_LENGTH,
        tasks=[task],
        probe_type=probe_type,
        batch_size=8,
    )

    mock_dataset = MockDataset(size=20, task_type=task_type)

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        # Set task type and num_classes
        callback.task_type = task_type
        if num_classes:
            callback.num_classes = num_classes

        mock_pl_module.device = "cpu"
        results = callback.evaluate(mock_pl_module, mock_trainer)

        # Verify results are generated
        assert isinstance(results, dict)

        # Check that probes were trained and stored
        assert len(callback.probes) > 0

        # Verify probe type matches expected sklearn classes
        for task_key, probe in callback.probes.items():
            if task_type == "regression":
                if probe_type == "linear":
                    from sklearn.linear_model import LinearRegression

                    assert isinstance(probe, LinearRegression)
                elif probe_type == "elastic":
                    from sklearn.linear_model import ElasticNet, LinearRegression

                    # Could be ElasticNet or LinearRegression (fallback)
                    assert isinstance(probe, (ElasticNet, LinearRegression))
                elif probe_type == "svm":
                    from sklearn.svm import SVR

                    assert isinstance(probe, SVR)
            else:  # multilabel
                from sklearn.multioutput import MultiOutputClassifier

                assert isinstance(probe, MultiOutputClassifier)


def test_combined_features(mock_trainer, mock_pl_module):
    """Test combination of cross-validation, dimensionality reduction, and different probe types."""
    callback = CalmLinearProbeCallback(
        max_length=MAX_LENGTH,
        tasks=["meltome"],
        use_cross_validation=True,
        n_folds=3,
        dimensionality_reduction=True,
        reduced_dim=16,
        probe_type="elastic",
        batch_size=8,
    )

    mock_dataset = MockDataset(size=30, task_type="regression")

    with patch("lobster.datasets._calm_property_dataset.CalmPropertyDataset", return_value=mock_dataset):
        callback.task_type = "regression"
        mock_pl_module.device = "cpu"

        results = callback.evaluate(mock_pl_module, mock_trainer)

        # Verify results are generated
        assert isinstance(results, dict)

        # Check that both dimensionality reduction and cross-validation worked
        # (dim_reducers should be created for each fold)
        assert len(callback.dim_reducers) > 0

        # Verify cross-validation metrics
        if "meltome" in results:
            task_metrics = results["meltome"]
            assert any("_std" in metric_name for metric_name in task_metrics.keys())
