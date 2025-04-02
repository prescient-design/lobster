import unittest.mock
from pathlib import Path

from datasets import Dataset
from pandas import DataFrame

from lobster.datasets import CalmDataset, CalmIterableDataset


class TestCalmDataset:
    """Unit tests for CalmDataset."""

    @unittest.mock.patch("lobster.datasets._calm_dataset.load_dataset")
    def test___init__(self, mock_load_dataset, tmp_path):
        """Test __init__ method."""
        mock_load_dataset.return_value = DataFrame({"sequence": ["ATG"], "description": ["dna"]})

        dataset = CalmDataset(root=tmp_path)

        item = dataset[0]

        assert item == ("ATG", "dna")

        assert dataset.root == Path(tmp_path).resolve()
        assert dataset.transform is None
        assert isinstance(dataset.data, DataFrame)


class TestCalmIterableDataset:
    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test__iter__(self, mock_load_dataset, tmp_path):
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "description": "dna",
                    "sequence": "atcg",
                }
            ]
        )
        dataset = CalmIterableDataset(
            keys=["sequence"], shuffle=False, download=False, root=tmp_path
        )  # root for caching
        example = next(iter(dataset))

        assert isinstance(example, str)
