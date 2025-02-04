import unittest.mock
from pathlib import Path
from tempfile import NamedTemporaryFile

from lobster.datasets import CalmDataset
from pandas import DataFrame


class TestCalmDataset:
    """Unit tests for CalmDataset."""

    @unittest.mock.patch("pandas.read_parquet")
    @unittest.mock.patch("pooch.retrieve")
    def test___init__(self, mock_retrieve, mock_read_parquet):
        """Test __init__ method."""
        with NamedTemporaryFile() as tmp:
            mock_retrieve.return_value = tmp.name
            mock_read_parquet.return_value = DataFrame({"sequence": ["ATG"], "description": ["dna"]})
            dataset = CalmDataset(root=tmp.name, download=True)

            assert dataset.root == Path(tmp.name).resolve()

            assert dataset.transform is None

            assert isinstance(dataset.data, DataFrame)
