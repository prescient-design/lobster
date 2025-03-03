import unittest.mock
from pathlib import Path
from tempfile import NamedTemporaryFile

from lobster.datasets import M320MDataset, M320MIterableDataset
from pandas import DataFrame


class TestM320MDataset:
    """Unit tests for M320MDataset."""

    @unittest.mock.patch("pandas.read_parquet")
    @unittest.mock.patch("pooch.retrieve")
    def test___init___(self, mock_retrieve, mock_read_parquet):
        with NamedTemporaryFile() as descriptor:
            mock_retrieve.return_value = descriptor.name
            mock_read_parquet.return_value = DataFrame({"smiles": ["C"], "Description": ["description"]})

            dataset = M320MDataset(descriptor.name, download=True)

            assert dataset.root == Path(descriptor.name).resolve()

            assert dataset.transform is None

            assert isinstance(dataset.data, DataFrame)


class TestAMPLIFYIterableDataset:
    def test__iter__(self):
        dataset = M320MIterableDataset(keys=["smiles"], shuffle=False, download=False)
        example = next(iter(dataset))

        assert isinstance(example, str)
