import pytest
from torch.utils.data import Dataset, IterableDataset

from lobster.datasets import MultiplexedSamplingDataset


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
        yield from self.data


@pytest.fixture
def datasets():
    return [
        IterableStringDataset(["Banana"] * 100),
        IterableStringDataset(["Apple"] * 500),
        IterableStringDataset(["Orange"] * 1000),
    ]


@pytest.fixture
def small_datasets():
    return [
        IterableStringDataset(["Banana"] * 3),
        IterableStringDataset(["Apple"] * 5),
        IterableStringDataset(["Orange"] * 7),
    ]


class TestMultiplexedSamplingDataset:
    def test__init__(self, datasets):
        # Test initialization with default weights
        dataset = MultiplexedSamplingDataset(datasets, seed=0)
        assert len(dataset.datasets) == 3
        assert pytest.approx(dataset.weights) == [1 / 3, 1 / 3, 1 / 3]
        assert dataset.seed == 0

        # Test initialization with custom weights
        custom_weights = [1.0, 2.0, 3.0]
        total = sum(custom_weights)
        dataset = MultiplexedSamplingDataset(datasets, weights=custom_weights, seed=0)
        assert pytest.approx(dataset.weights) == [w / total for w in custom_weights]

        # Test mismatched weights
        with pytest.raises(ValueError):
            MultiplexedSamplingDataset(datasets, weights=[1.0, 2.0])

    def test_different_seed_different_samples(self, datasets):
        # Test that different seeds produce different sampling sequences
        dataset1 = MultiplexedSamplingDataset(datasets, seed=42)
        dataset2 = MultiplexedSamplingDataset(datasets, seed=43)

        samples1 = []
        samples2 = []

        # Collect first 100 samples from each dataset
        for i, sample in enumerate(dataset1):
            samples1.append(sample)
            if i >= 99:
                break

        for i, sample in enumerate(dataset2):
            samples2.append(sample)
            if i >= 99:
                break

        # Count occurrences of each fruit
        banana_count1 = samples1.count("Banana")
        banana_count2 = samples2.count("Banana")

        # With different seeds, we expect different distributions
        # This is probabilistic, but very likely to be different with 100 samples
        assert banana_count1 != banana_count2

    def test_same_seed_same_samples(self, datasets):
        # Test that same seeds produce identical sampling sequences
        dataset1 = MultiplexedSamplingDataset(datasets, seed=42)
        dataset2 = MultiplexedSamplingDataset(datasets, seed=42)

        samples1 = []
        samples2 = []

        # Collect first 50 samples from each dataset
        for i, sample in enumerate(dataset1):
            samples1.append(sample)
            if i >= 49:
                break

        for i, sample in enumerate(dataset2):
            samples2.append(sample)
            if i >= 49:
                break

        # With the same seed, samples should be identical
        assert samples1 == samples2

    def test_weighted_sampling_extreme(self, datasets):
        # Test with extreme weights (99% from one dataset)
        weights = [0.99, 0.005, 0.005]
        dataset = MultiplexedSamplingDataset(datasets, weights=weights, seed=42)

        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            if i >= 199:
                break

        # Count occurrences
        banana_count = samples.count("Banana")

        # With extreme weights, expect close to 99% bananas
        assert 0.95 <= banana_count / len(samples) <= 1.0

    def test_empty_dataset(self):
        # Test with one empty dataset
        datasets = [
            IterableStringDataset([]),
            IterableStringDataset(["Apple"] * 5),
        ]

        dataset = MultiplexedSamplingDataset(datasets, seed=42)

        # First yield should trigger StopIteration for the empty dataset
        # which should cause the entire iteration to stop
        samples = list(dataset)

        # Should stop after first empty dataset encountered
        assert len(samples) == 0

    def test_non_positive_weights(self, datasets):
        # Test with zero or negative weights
        with pytest.raises(ValueError):
            # Random.choices doesn't accept negative weights
            MultiplexedSamplingDataset(datasets, weights=[-1.0, 1.0, 1.0], seed=42)

        # Zero weights should work but won't sample from that dataset
        dataset = MultiplexedSamplingDataset(datasets, weights=[0.0, 1.0, 1.0], seed=42)

        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            if i >= 99:
                break

        # Should not have any bananas (first dataset has zero weight)
        assert "Banana" not in samples
        assert "Apple" in samples
        assert "Orange" in samples
