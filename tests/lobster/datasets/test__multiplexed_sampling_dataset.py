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
        i = 0

        while i < len(self.data):
            yield self.data[i]
            i += 1


@pytest.fixture
def datasets():
    return [
        StringDataset(["1"]),
        StringDataset(
            [
                "Apple",
                "Orange",
            ]
        ),
        IterableStringDataset(
            [
                "A",
                "B",
                "C",
            ]
        ),
    ]


class TestMultiplexedSamplingDataset:
    def test__init__(self, datasets):
        dataset = MultiplexedSamplingDataset(datasets, seed=0, mode="min")

        assert len(dataset.datasets) == 3
        assert pytest.approx(dataset.weights) == [1 / 3, 1 / 3, 1 / 3]
        assert dataset.seed == 0

    def test__iter__min(self, datasets):
        weights = [100, 1, 1]
        dataset = MultiplexedSamplingDataset(datasets, weights, seed=0, mode="min")

        samples = list(dataset)
        assert samples == ["1"]

    @pytest.mark.parametrize(
        "weights",
        [
            pytest.param([1, 1, 1], id="equal_weights"),
        ],
    )
    def test__iter__max_size_cycle(self, weights, datasets):
        dataset = MultiplexedSamplingDataset(datasets, weights, seed=0, mode="max_size_cycle")

        samples = list(dataset)

        assert set(samples) == {"1", "A", "Apple", "Orange", "B", "C"}
