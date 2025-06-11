import unittest.mock

from datasets import Dataset

from lobster.datasets import CalmIterableDataset


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
