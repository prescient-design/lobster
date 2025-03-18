import unittest.mock

from datasets import Dataset

from lobster.datasets import LatentGeneratorPinderIterableDataset


class TestLatentGeneratorPinderIterableDataset:
    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test__iter__(self, mock_load_dataset):
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "__index_level_0__": "7u9z__G1_Q92736--7u9z__D1_P68106",
                    "lg_token_string": [
                        "ft ec ec hp ek bt bt ek . da da ek da da ec da hx ec",
                        "ec ec ec da da da hp hp hp . bt da ig ig ig ig da gv hp",
                        "gj dp gj c gj ec hx cd cz cg . ec ec gj if da ft fe ft",
                        "fe l hx hx da da hx hx hx hx hx . hx hx da ec ec da ft",
                    ],
                }
            ]
        )

        dataset = LatentGeneratorPinderIterableDataset(shuffle=False, download=False)
        example = next(iter(dataset))

        assert isinstance(example, list)
        assert "|" not in example[0]
        assert "." in example[0]
