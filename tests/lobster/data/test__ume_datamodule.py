import unittest.mock
from pathlib import Path

import pytest
from datasets import Dataset
from torch import Tensor
from torch.utils.data import DataLoader

from lobster.constants import Modality
from lobster.data import UMELightningDataModule
from lobster.datasets import MultiplexedSamplingDataset


@pytest.fixture
def dm(tmp_path):
    dm = UMELightningDataModule(root=tmp_path, datasets=["AMPLIFY"], batch_size=8, tokenizer_max_length=512)
    return dm


class TestUMELightningDataModule:
    def test__init__(self, dm):
        assert dm._batch_size == 8
        assert dm._tokenizer_max_length == 512
        assert isinstance(dm._root, str | Path)

    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test_train_dataloader(self, mock_load_dataset, dm):
        # Create enough mock data for a batch
        mock_data = []
        for i in range(20):  # Create 20 samples to ensure we have enough for a batch
            mock_data.append(
                {
                    "sequence": f"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG{i}",
                    "name": f"test_protein_{i}",
                }
            )

        mock_load_dataset.return_value = Dataset.from_list(mock_data)

        dm.setup()

        dataloader = dm.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8

        batch = next(iter(dataloader))
        assert isinstance(batch["input_ids"], Tensor)
        assert isinstance(batch["attention_mask"], Tensor)
        assert isinstance(batch["modality"], list)
        assert isinstance(batch["modality"][0], Modality)

    @unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset")
    def test_train_dataloader_multiplex(self, mock_load_dataset, tmp_path):
        # Create enough mock data for a batch
        mock_data = []
        for i in range(20):  # Create 20 samples to ensure we have enough for a batch
            mock_data.append(
                {
                    "sequence": f"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG{i}",
                    "name": f"test_protein_{i}",
                }
            )

        mock_load_dataset.return_value = Dataset.from_list(mock_data)

        dm = UMELightningDataModule(
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
