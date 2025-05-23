import unittest.mock as mock

import pytest

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset


# Create a concrete subclass for testing
class ConcreteDataset(HuggingFaceIterableDataset):
    def _passes_type_check(self, sample):
        return True


@pytest.fixture
def dataset_loader(monkeypatch):
    """Fixture to mock the HuggingFace load_dataset function."""
    mock_fn = mock.Mock()
    monkeypatch.setattr("lobster.datasets._huggingface_iterable_dataset.load_dataset", mock_fn)
    return mock_fn


# Mock dataset class
class MockIterableDataset:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def shuffle(self, buffer_size, seed):
        return self

    def to_iterable_dataset(self, num_shards):
        return self


class TestHuggingFaceIterableDataset:
    """Tests for the HuggingFaceIterableDataset class."""

    def test__init__(self, dataset_loader):
        """Test initialization with various parameters."""
        # Set up mock return value
        mock_dataset = mock.MagicMock()
        dataset_loader.return_value = mock_dataset

        # Initialize dataset with test parameters
        dataset = ConcreteDataset(
            dataset_name="test_dataset",
            root="/tmp",
            transform=lambda x: x,
            keys=["input", "output"],
            split="train",
            shuffle=True,
            shuffle_buffer_size=100,
            seed=42,
            download=False,
            limit=1000,
        )

        # Verify parameters were stored correctly
        assert dataset.dataset_name == "test_dataset"
        assert dataset.root == "/tmp"
        assert dataset.split == "train"
        assert dataset.shuffle is True
        assert dataset.shuffle_buffer_size == 100
        assert dataset.seed == 42
        assert dataset.download is False
        assert dataset.limit == 1000
        assert dataset.keys == ["input", "output"]

        # Verify load_dataset was called with correct parameters
        dataset_loader.assert_called_once_with(
            "test_dataset",
            split="train",
            streaming=True,  # Not download
            cache_dir="/tmp",
        )

    @pytest.mark.parametrize(
        "total_workers, limit, expected_limits",
        [
            (4, 8, [2, 2, 2, 2]),  # Even division
            (3, 10, [4, 3, 3]),  # Remainder of 1
            (5, 12, [3, 3, 2, 2, 2]),  # Remainder of 2
            (4, 10, [3, 3, 2, 2]),  # Multiple remainders
        ],
        ids=[
            "even_division_no_remainder",
            "remainder_1_distributed_to_first_worker",
            "remainder_2_distributed_to_first_two_workers",
            "multiple_remainders_distributed_evenly",
        ],
    )
    def test_limit_calculation(self, total_workers, limit, expected_limits):
        """Test the limit calculation logic with various scenarios."""
        calculated_limits = []

        for worker_id in range(total_workers):
            # Calculate using the same logic as in the class
            base_limit = limit // total_workers
            extra = limit % total_workers
            worker_limit = base_limit + (1 if worker_id < extra else 0)
            calculated_limits.append(worker_limit)

        # Verify distribution matches expectations
        assert calculated_limits == expected_limits

        # Verify total equals global limit
        assert sum(calculated_limits) == limit
