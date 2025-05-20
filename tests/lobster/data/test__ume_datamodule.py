from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lobster.constants import Modality
from lobster.data import UmeLightningDataModule
from lobster.datasets import MultiplexedSamplingDataset
from torch import Tensor
from torch.utils.data import DataLoader


@pytest.fixture
def dm(tmp_path):
    dm = UmeLightningDataModule(root=tmp_path, datasets=["AMPLIFY"], batch_size=8, tokenizer_max_length=512)
    return dm


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
        dm = UmeLightningDataModule(
            root=tmp_path,
            datasets=["AMPLIFY", "Calm"],
            batch_size=8,
            tokenizer_max_length=512,
            sample=True,
            weights=None,
        )
        dm.setup()

        expected_weights = [w / sum(dm._train_sizes) for w in dm._train_sizes]

        assert dm._sample
        assert isinstance(dm.train_dataset, MultiplexedSamplingDataset)
        assert dm.train_dataset.weights == expected_weights

        dataloader = dm.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8

        batch = next(iter(dataloader))
        assert isinstance(batch, dict)
        assert isinstance(batch["input_ids"], Tensor)
        assert isinstance(batch["attention_mask"], Tensor)
        assert isinstance(batch["modality"], list)
        assert isinstance(batch["modality"][0], Modality)

    @patch("lobster.data.ume_datamodule.UmeDataset")
    def test_setup_with_mocked_dataset(self, mock_dataset, tmp_path):
        # Create mock dataset instances
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 100
        mock_dataset.return_value = mock_dataset_instance

        # Configure return values
        mock_dataset_instance.data = [
            {"input_ids": Tensor([1]), "attention_mask": Tensor([1]), "modality": Modality.TEXT}
        ]

        # Create datamodule with mock
        dm = UmeLightningDataModule(root=tmp_path, datasets=["AMPLIFY"], batch_size=8, tokenizer_max_length=512)

        # Setup the datamodule
        dm.setup()

        # Verify dataset was created with correct parameters
        mock_dataset.assert_called_once()
        assert mock_dataset.call_args[1]["dataset_name"] == "AMPLIFY"

        # Verify dataloader works with mocked dataset
        dataloader = dm.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8
