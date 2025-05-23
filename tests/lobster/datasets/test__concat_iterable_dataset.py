from collections import Counter

import pytest
from torch.utils.data import IterableDataset

from lobster.datasets import RoundRobinConcatIterableDataset


class IterableStringDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        yield from self.data


@pytest.fixture
def datasets():
    return [
        IterableStringDataset(["Banana"] * 10),
        IterableStringDataset(["Apple"] * 20),
        IterableStringDataset(["Orange"] * 30),
    ]


class TestRoundRobinConcatIterableDataset:
    def test__init__(self, datasets):
        # Test initialization
        dataset = RoundRobinConcatIterableDataset(datasets)
        assert len(dataset.datasets) == 3
        assert dataset.datasets[0] is datasets[0]
        assert dataset.datasets[1] is datasets[1]
        assert dataset.datasets[2] is datasets[2]

    def test_iteration_order_max(self, datasets):
        dataset = RoundRobinConcatIterableDataset(datasets, stopping_condition="max")

        samples = list(dataset)
        count = Counter(samples)

        assert count["Banana"] == 10
        assert count["Apple"] == 20
        assert count["Orange"] == 30

    def test_iteration_order_min(self, datasets):
        dataset = RoundRobinConcatIterableDataset(datasets, stopping_condition="min")

        samples = list(dataset)
        count = Counter(samples)

        assert count["Banana"] == 10
        assert count["Apple"] == 10
        assert count["Orange"] == 10

    def test_iteration_order_min_single_sample(self):
        datasets = [
            IterableStringDataset(["Banana"] * 1),
            IterableStringDataset(["Apple"] * 20),
            IterableStringDataset(["Orange"] * 30),
        ]
        dataset = RoundRobinConcatIterableDataset(datasets, stopping_condition="min")

        samples = list(dataset)
        count = Counter(samples)

        assert count["Banana"] == 1
        assert count["Apple"] == 1
        assert count["Orange"] == 1
