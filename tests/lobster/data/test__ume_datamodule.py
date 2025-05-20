from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lobster.constants import Modality
from lobster.data import UmeLightningDataModule
from lobster.datasets import AMPLIFYIterableDataset
from torch import Tensor
from torch.utils.data import DataLoader


@pytest.fixture
def dm(tmp_path):
    # Create mock data for the dataset to return
    mock_batch = {
        "input_ids": Tensor([[1, 2, 3]]),
        "attention_mask": Tensor([[1, 1, 1]]),
        "modality": [Modality.AMINO_ACID],
    }

    # Create a mock dataset that returns our mock data
    mock_dataset = MagicMock(spec=AMPLIFYIterableDataset)
    mock_dataset.__iter__.return_value = iter([mock_batch])

    # Patch the dataset class to return our controlled mock
    with patch("lobster.datasets.AMPLIFYIterableDataset", return_value=mock_dataset):
        return UmeLightningDataModule(root=tmp_path, datasets=["AMPLIFY"], batch_size=8, tokenizer_max_length=512)


class TestUmeLightningDataModule:
    def test__init__(self, dm):
        assert dm._batch_size == 8
        assert dm._tokenizer_max_length == 512
        assert isinstance(dm._root, str | Path)

    def test_train_dataloader(self, dm):
        dm.setup()

        dataloader = dm.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8

        batch = next(iter(dataloader))
        assert isinstance(batch["input_ids"], Tensor)
        assert isinstance(batch["attention_mask"], Tensor)
        assert isinstance(batch["modality"], list)
        assert isinstance(batch["modality"][0], Modality)

    def test_train_dataloader_multiplex(self, tmp_path):
        # Create mock datasets for both AMPLIFY and Calm
        mock_amplify = MagicMock(spec=AMPLIFYIterableDataset)
        mock_calm = MagicMock(spec=AMPLIFYIterableDataset)

        # Set up mock data
        mock_batch = {
            "input_ids": Tensor([[1, 2, 3]]),
            "attention_mask": Tensor([[1, 1, 1]]),
            "modality": [Modality.AMINO_ACID],
        }

        # Configure mocks to return our test data
        mock_amplify.__iter__.return_value = iter([mock_batch])
        mock_calm.__iter__.return_value = iter([mock_batch])

        # Mock train sizes for weight calculation
        mock_amplify.train_size = 100
        mock_calm.train_size = 200

        # Patch both dataset classes
        with (
            patch("lobster.datasets.AMPLIFYIterableDataset") as mock_amplify_cls,
            patch("lobster.datasets.CalmIterableDataset") as mock_calm_cls,
        ):
            mock_amplify_cls.return_value = mock_amplify
            mock_calm_cls.return_value = mock_calm

            dm = UmeLightningDataModule(
                root=tmp_path,
                datasets=["AMPLIFY", "Calm"],
                batch_size=8,
                tokenizer_max_length=512,
                sample=True,
                weights=None,
            )
            dm.setup()

            dataloader = dm.train_dataloader()
            assert isinstance(dataloader, DataLoader)
            assert dataloader.batch_size == 8

            batch = next(iter(dataloader))
            assert isinstance(batch, dict)
            assert isinstance(batch["input_ids"], Tensor)
            assert isinstance(batch["attention_mask"], Tensor)
            assert isinstance(batch["modality"], list)
            assert isinstance(batch["modality"][0], Modality)
