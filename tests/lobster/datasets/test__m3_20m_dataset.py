import unittest.mock

from datasets import Dataset

from lobster.datasets import M320MIterableDataset


class TestM320MIterableDataset:
    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test__iter__(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "Description": "text",
                    "smiles": "CCCO",
                }
            ]
        )
        dataset = M320MIterableDataset(keys=["smiles"], shuffle=False, download=False)
        example = next(iter(dataset))

        assert isinstance(example, str)
