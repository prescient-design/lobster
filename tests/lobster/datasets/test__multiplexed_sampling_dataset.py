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

        # Check total count
        assert len(samples) == 2000

    def test_weighted_sampling_with_max_size(self, datasets):
        # Test sampling with custom weights and fixed max_size
        dataset = MultiplexedSamplingDataset(datasets, weights=[100, 500, 1000], seed=0, max_size=2000)
        samples = list(dataset)

        # Check total count
        assert len(samples) == 2000
