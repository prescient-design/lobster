from collections import Counter

import pytest
from lobster.datasets import MultiplexedSamplingDataset
from torch.utils.data import Dataset, IterableDataset


class StringDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IterableStringDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for item in self.data:
            yield item


@pytest.fixture
def datasets():
    return [
        IterableStringDataset(["Banana"] * 100),
        IterableStringDataset(["Apple"] * 500),
        IterableStringDataset(["Orange"] * 1000),
    ]


class TestMultiplexedSamplingDataset:
    def test__init__(self, datasets):
        # Test initialization with default weights
        dataset = MultiplexedSamplingDataset(datasets, seed=0, mode="min")
        assert len(dataset.datasets) == 3
        assert pytest.approx(dataset.weights.tolist()) == [1 / 3, 1 / 3, 1 / 3]
        assert dataset.seed == 0
        assert dataset.mode == "min"

        # Test initialization with custom weights
        custom_weights = [1.0, 2.0, 3.0]
        total = sum(custom_weights)
        dataset = MultiplexedSamplingDataset(datasets, weights=custom_weights, seed=0)
        assert pytest.approx(dataset.weights.tolist()) == [w / total for w in custom_weights]

        # Test invalid mode
        with pytest.raises(ValueError):
            MultiplexedSamplingDataset(datasets, mode="invalid_mode")

        # Test mismatched weights
        with pytest.raises(ValueError):
            MultiplexedSamplingDataset(datasets, weights=[1.0, 2.0])

    def test_equal_sampling_with_max_size(self, datasets):
        # Test sampling with equal probability and fixed max_size
        dataset = MultiplexedSamplingDataset(datasets, seed=0, max_size=2000)
        samples = list(dataset)
        counts = Counter(samples)

        # Check total count
        assert len(samples) == 2000

        # With seed=0, we should get exact counts since it's deterministic
        assert counts["Orange"] == 711
        assert counts["Banana"] == 648
        assert counts["Apple"] == 641

    def test_weighted_sampling_with_max_size(self, datasets):
        # Test sampling with custom weights and fixed max_size
        dataset = MultiplexedSamplingDataset(datasets, weights=[100, 500, 1000], seed=0, max_size=2000)
        samples = list(dataset)
        counts = Counter(samples)

        # Check total count
        assert len(samples) == 2000

        # With seed=0, we should get exact counts since it's deterministic
        assert counts["Banana"] == 103  # ~6.25% (100/1600)
        assert counts["Apple"] == 610  # ~31.25% (500/1600)
        assert counts["Orange"] == 1287  # ~62.5% (1000/1600)

    def test_min_mode(self, datasets):
        # Test sampling with min mode (stops after shortest dataset)
        dataset = MultiplexedSamplingDataset(datasets, seed=0, mode="min")
        samples = list(dataset)
        counts = Counter(samples)

        # From the docstring example with seed=0
        assert len(samples) == 304  # Total count from the example
        assert counts["Orange"] == 106
        assert counts["Banana"] == 100
        assert counts["Apple"] == 98

    def test_max_size_cycle_mode(self, datasets):
        # Test sampling with max_size_cycle mode
        dataset = MultiplexedSamplingDataset(datasets, seed=0, mode="max_size_cycle")
        samples = list(dataset)
        counts = Counter(samples)

        # From the docstring example with seed=0
        assert len(samples) == 2838  # Total count from the example
        assert counts["Orange"] == 1000
        assert counts["Banana"] == 925  # Cycled
        assert counts["Apple"] == 913  # Cycled
