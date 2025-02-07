import pytest
from lobster.datasets import MultiplexedDataset
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


class Test_MultiplexedDataset:
    def test__init__(self, datasets):
        dataset = MultiplexedDataset(datasets, seed=0, mode="min")

        assert len(dataset.datasets) == 3
        assert pytest.approx(dataset.weights) == [1 / 3, 1 / 3, 1 / 3]
        assert dataset.seed == 0

    def test__iter__min(self, datasets):
        weights = [100, 1, 1]
        dataset = MultiplexedDataset(datasets, weights, seed=0, mode="min")

        samples = list(iter(dataset))
        assert samples == ["1"]

    # TODO zadorozk: fix this, it's not deterministic even with seed
    # @pytest.mark.parametrize(
    #     "weights, expected_samples",
    #     [
    #         pytest.param([1, 1, 1], ["A", "B", "Apple", "1", "Orange", "C", "Apple", "Orange"], id="equal_weights"),
    #     ],
    # )
    # def test__iter__max_size_cycle(self, weights, expected_samples, datasets):
    #     dataset = MultiplexedDataset(datasets, weights, seed=0, mode="max_size_cycle")

    #     samples = list(iter(dataset))

    #     assert samples == expected_samples
