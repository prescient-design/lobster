from unittest.mock import Mock, patch

import pytest
from torch.utils.data import IterableDataset

from lobster.datasets import ShuffledIterableDataset


class SimpleIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data


@pytest.fixture
def dataset():
    return SimpleIterableDataset(range(100))


class TestShuffledIterableDataset:
    def test_init(self, dataset):
        dataset = ShuffledIterableDataset(dataset, buffer_size=100, seed=42)
        assert dataset.buffer_size == 100
        assert dataset.seed == 42

    def test_basic_iteration(self, dataset):
        shuffled = ShuffledIterableDataset(dataset, buffer_size=10)
        items = list(shuffled)

        assert len(items) == 100
        assert set(items) == set(range(100))

        # Check items are shuffled (not in original order)
        assert items != list(range(100))

    def test_seed_reproducibility(self, dataset):
        """Test that same seed produces same shuffle."""
        seed = 42
        shuffled1 = ShuffledIterableDataset(dataset, buffer_size=10, seed=seed)
        items1 = list(shuffled1)

        shuffled2 = ShuffledIterableDataset(dataset, buffer_size=10, seed=seed)
        items2 = list(shuffled2)

        assert items1 == items2

    def test_different_seeds(self, dataset):
        """Test that different seeds produce different shuffles."""
        shuffled1 = ShuffledIterableDataset(dataset, buffer_size=10, seed=42)
        items1 = list(shuffled1)

        shuffled2 = ShuffledIterableDataset(dataset, buffer_size=10, seed=43)
        items2 = list(shuffled2)

        assert items1 != items2

    @patch("lobster.datasets._shuffled_iterable_dataset.get_worker_info")
    def test_worker_seed_handling(self, mock_get_worker_info, dataset):
        """Test that different workers get different but deterministic seeds."""
        # Mock worker info
        mock_worker = Mock()
        mock_worker.id = 1
        mock_get_worker_info.return_value = mock_worker

        seed = 42
        shuffled = ShuffledIterableDataset(dataset, buffer_size=10, seed=seed)
        items_worker1 = list(shuffled)

        # Change worker ID
        mock_worker.id = 2
        shuffled = ShuffledIterableDataset(dataset, buffer_size=10, seed=seed)
        items_worker2 = list(shuffled)

        # Different workers should get different but deterministic sequences
        assert items_worker1 != items_worker2

        # Same worker should get same sequence
        mock_worker.id = 1
        shuffled = ShuffledIterableDataset(dataset, buffer_size=10, seed=seed)
        items_worker1_repeat = list(shuffled)
        assert items_worker1 == items_worker1_repeat
