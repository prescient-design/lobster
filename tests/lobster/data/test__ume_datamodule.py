import unittest.mock

import pytest
from lobster.data import UmeLightningDataModule
from lobster.datasets import MultiplexedDataset
from pandas import DataFrame
from torch import Tensor
from transformers import BatchEncoding


@pytest.fixture
def dm(tmp_path):
    with unittest.mock.patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = DataFrame({"smiles": 10 * ["CCO"]})

        datamodule = UmeLightningDataModule(root=tmp_path, datasets=["ChEMBL"], batch_size=8, tokenizer_max_length=512)
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule


class TestUmeLightningDataModule:
    def test_prepare_data(self, dm: UmeLightningDataModule):
        assert isinstance(dm.dataset, MultiplexedDataset)

    def test_train_dataloader(self, dm: UmeLightningDataModule):
        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))

        assert isinstance(batch, BatchEncoding)
        assert isinstance(batch["input_ids"], Tensor)
