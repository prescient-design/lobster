import pytest
from lobster.datasets import M320MDataset


@pytest.fixture(autouse=True)
def dataset() -> M320MDataset:
    return M320MDataset()


class TestM320MDataset:
    """Unit tests for M320MDataset."""

    def test___init___subset(self, dataset: M320MDataset):
        item0 = dataset[0]
        assert len(item0) == 2
        assert isinstance(item0[0], str)
        assert isinstance(item0[1], str)
        assert len(item0[1]) > 50

    @pytest.mark.skip(reason="Requires download.")
    def test___init___full(self, tmp_path):
        (tmp_path / "m320m").mkdir(parents=True, exist_ok=True)

        dataset = M320MDataset(root=tmp_path, download=True, full_dataset=True)

        assert dataset.root.exists()

        item0 = dataset[0]
        assert len(item0) == 2
        assert isinstance(item0[0], str)
        assert isinstance(item0[1], str)
        assert len(item0[1]) > 50
