import pytest
from lobster.datasets import CalmDataset


@pytest.fixture(autouse=True)
def dataset(tmp_path):
    (tmp_path / "calm").mkdir(parents=True, exist_ok=True)

    return CalmDataset(root=tmp_path, download=True, train=False)


@pytest.mark.skip(reason="Requires download.")
class TestCalmDataset:
    """Unit tests for CalmDataset."""

    def test___init__(self, dataset: CalmDataset):
        assert dataset.root.exists()

        item0 = dataset[0]
        assert len(item0) == 2
        assert isinstance(item0[0], str)
        assert len(dataset) == 4351
