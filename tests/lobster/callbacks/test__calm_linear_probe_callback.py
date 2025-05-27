from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from lobster.callbacks import CalmLinearProbeCallback
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
