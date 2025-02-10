import unittest.mock
from tempfile import NamedTemporaryFile

import pytest
from lobster.datasets import M320MDataset
from pandas import DataFrame


class TestM320MDataset:
    """Unit tests for M320MDataset."""

    @unittest.mock.patch("pandas.read_parquet")
    @unittest.mock.patch("pooch.retrieve")
    @pytest.fixture
    def dataset(mock_retrieve, mock_read_parquet):
        with NamedTemporaryFile() as descriptor:
            mock_retrieve.return_value = descriptor.name
            mock_read_parquet.return_value = DataFrame({"smiles": ["C"], "Description": ["description"]})

            return M320MDataset(descriptor.name, download=True)

    def test__iter__(self):
        dataset = M320MDataset(
            root=None,
            download=False,
            transform=None,
            columns=None,
        )

        item = next(iter(dataset))

        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], str)
