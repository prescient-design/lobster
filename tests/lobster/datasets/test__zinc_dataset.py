import unittest.mock

from datasets import Dataset

from lobster.datasets import ZINCIterableDataset


class TestZINCIterableDataset:
    """Unit tests for ZINCIterableDataset."""

    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test__init__(self, mock_load_dataset):
        """Test that initialization sets attributes correctly."""
        mock_dataset = unittest.mock.MagicMock()
        mock_load_dataset.return_value = mock_dataset

        dataset = ZINCIterableDataset(
            root="/test/path", keys=["smiles", "selfies"], split="validation", shuffle=False, limit=100
        )

        # Assert initialization parameters are stored correctly
        assert dataset.dataset_name == "haydn-jones/ZINC20"
        assert dataset.root == "/test/path"
        assert dataset.keys == ["smiles", "selfies"]
        assert dataset.split == "validation"
        assert dataset.shuffle is False
        assert dataset.limit == 100

        # Verify load_dataset was called with correct parameters
        mock_load_dataset.assert_called_once_with(
            "haydn-jones/ZINC20",
            split="validation",
            streaming=True,  # Not download
            cache_dir="/test/path",
        )

    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test__iter__(self, mock_load_dataset):
        """Test iteration and type checking."""
        # Mock dataset with sample zinc data
        mock_load_dataset.return_value = Dataset.from_list(
            [
                {
                    "smiles": "CCCCOc1ccc(C(=O)N2CC[C@H](N[C@H](C)COC)C2)cc1",
                    "zinc_id": 1266565665,
                    "selfies": "[C][C][C][C][O][C][=C][C][=C][Branch2][Ring1][=Branch1][C][=Branch1][C][=O][N][C][C][C@H1]",
                }
            ]
        )

        # Test with multiple keys
        dataset = ZINCIterableDataset(keys=["smiles", "selfies"], shuffle=False)
        example = next(iter(dataset))

        # Check example is a tuple of strings
        assert isinstance(example, tuple)
        assert len(example) == 2
        assert all(isinstance(item, str) for item in example)
        assert example[0] == "CCCCOc1ccc(C(=O)N2CC[C@H](N[C@H](C)COC)C2)cc1"
        assert example[1].startswith("[C][C][C][C][O][C]")
