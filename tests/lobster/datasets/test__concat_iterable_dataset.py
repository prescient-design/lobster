import pytest
from lobster.datasets import ConcatIterableDataset
from torch.utils.data import IterableDataset


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
        IterableStringDataset(["Apple"] * 200),
        IterableStringDataset(["Orange"] * 300),
    ]


@pytest.fixture
def small_datasets():
    return [
        IterableStringDataset(["Banana"] * 3),
        IterableStringDataset(["Apple"] * 5),
        IterableStringDataset(["Orange"] * 7),
    ]


class TestConcatIterableDataset:
    def test__init__(self, datasets):
        # Test initialization
        dataset = ConcatIterableDataset(datasets)
        assert len(dataset.datasets) == 3
        assert dataset.datasets[0] is datasets[0]
        assert dataset.datasets[1] is datasets[1]
        assert dataset.datasets[2] is datasets[2]

    def test_iteration_order(self, small_datasets):
        # Test iteration order - should yield one item from each dataset in sequence
        dataset = ConcatIterableDataset(small_datasets)

        # Get first 10 samples
        first_samples = []
        for i, sample in enumerate(dataset):
            first_samples.append(sample)
            if i >= 9:
                break

        # Check the expected pattern: [Banana, Apple, Orange, Banana, Apple, Orange, ...]
        assert first_samples[0] == "Banana"
        assert first_samples[1] == "Apple"
        assert first_samples[2] == "Orange"
        assert first_samples[3] == "Banana"
        assert first_samples[4] == "Apple"
        assert first_samples[5] == "Orange"

    def test_dataset_exhaustion(self, small_datasets):
        # Test behavior when datasets are exhausted one by one
        dataset = ConcatIterableDataset(small_datasets)

        # Collect all samples
        all_samples = list(dataset)

        # Check total count (3 + 5 + 7 = 15)
        assert len(all_samples) == 15

        # Count occurrences of each fruit
        banana_count = all_samples.count("Banana")
        apple_count = all_samples.count("Apple")
        orange_count = all_samples.count("Orange")

        # Verify counts match the input datasets
        assert banana_count == 3
        assert apple_count == 5
        assert orange_count == 7

        # The last samples should be "Orange" since it's the last dataset to be exhausted
        assert all_samples[-1] == "Orange"

    def test_empty_datasets(self):
        # Test with empty datasets
        empty_datasets = [
            IterableStringDataset([]),
            IterableStringDataset(["Apple"] * 3),
            IterableStringDataset([]),
        ]

        dataset = ConcatIterableDataset(empty_datasets)
        samples = list(dataset)

        # Should only yield items from the non-empty dataset
        assert len(samples) == 3
        assert all(sample == "Apple" for sample in samples)

    def test_single_dataset(self):
        # Test with a single dataset
        single_dataset = [IterableStringDataset(["Mango"] * 5)]
        dataset = ConcatIterableDataset(single_dataset)

        samples = list(dataset)
        assert len(samples) == 5
        assert all(sample == "Mango" for sample in samples)

    def test_no_datasets(self):
        # Test with no datasets
        dataset = ConcatIterableDataset([])
        samples = list(dataset)
        assert len(samples) == 0
