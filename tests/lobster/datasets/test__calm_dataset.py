import pytest
from lobster.datasets import CalmDataset


@pytest.fixture(autouse=True)
def dataset(tmp_path):
    (tmp_path / "calm").mkdir(parents=True, exist_ok=True)

    return CalmDataset(root=tmp_path, download=True, train=False)


class TestCalmDataset:
    """Unit tests for CalmDataset."""

    def test___init__(self, dataset: CalmDataset):
        assert dataset._root.exists()

    def test___getitem__(self, dataset: CalmDataset):
        item0 = dataset[0]
        assert len(item0) == 2142
        assert isinstance(item0, str)

    def test___len__(self, dataset: CalmDataset):
        assert len(dataset) == 528
