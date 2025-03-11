import unittest.mock

from datasets import Dataset

from lobster.datasets import AMPLIFYIterableDataset


class TestAMPLIFYIterableDataset:
    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test__iter__(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "name": ">cf9945fa-34f9-4676-bd69-d61715a195a3|_A",
                    "sequence": "QVQLVQSGAEVKKPGASVKV|SCKASGYTFTGYYMHWVRQAP",
                }
            ]
        )

        dataset = AMPLIFYIterableDataset(shuffle=False, download=False)
        example = next(iter(dataset))

        assert isinstance(example, str)
        assert "|" not in example
        assert "." in example
