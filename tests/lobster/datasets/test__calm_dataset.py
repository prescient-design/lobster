import unittest.mock
from pathlib import Path

from lobster.datasets import CalmDataset
from pandas import DataFrame


class TestCalmDataset:
    """Unit tests for CalmDataset."""

    @unittest.mock.patch("lobster.datasets._calm_dataset.load_dataset")
    def test___init__(self, mock_load_dataset, tmp_path):
        """Test __init__ method."""
        mock_load_dataset.return_value = DataFrame({"sequence": ["ATG"], "description": ["dna"]})

        dataset = CalmDataset(root=tmp_path, download=False)

        item = dataset[0]

        assert item == ("ATG", "dna")

        assert dataset.root == Path(tmp_path).resolve()
        assert dataset.transform is None
        assert isinstance(dataset.data, DataFrame)
