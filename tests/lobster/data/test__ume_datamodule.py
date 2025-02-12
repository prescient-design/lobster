import unittest.mock
from pathlib import Path

import pytest
from lobster.data import UmeLightningDataModule
from lobster.datasets import MultiplexedSamplingDataset
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import ChainDataset, DataLoader
from transformers import BatchEncoding


@pytest.fixture
def mock_smiles_data():
    return DataFrame({"smiles": 10 * ["CCO"]})


@pytest.fixture
def dm(tmp_path):
    dm = UmeLightningDataModule(root=tmp_path, datasets=["ChEMBL"], batch_size=8, tokenizer_max_length=512)
    return dm


@pytest.fixture
def dm_weighted(tmp_path):
    dm = UmeLightningDataModule(
        root=tmp_path, datasets={"ChEMBL": 2}, enable_sampling=True, batch_size=8, tokenizer_max_length=512
    )
    return dm


class TestUmeLightningDataModule:
    def test__init__(self, dm):
        assert dm.weights is None
        assert dm._batch_size == 8
        assert dm._tokenizer_max_length == 512
        assert isinstance(dm._root, (str, Path))

    def test__init__weighted(self, dm_weighted):
        assert dm_weighted.weights == [2]
        assert dm_weighted._enable_sampling is True

    def test_init_invalid_dataset(self, tmp_path):
        with pytest.raises(ValueError, match="Only the following datasets"):
            UmeLightningDataModule(root=tmp_path, datasets=["InvalidDataset"], tokenizer_max_length=512)

    def test_init_weights_without_sampling(self, tmp_path):
        with pytest.raises(ValueError, match="Weights can only be provided"):
            UmeLightningDataModule(root=tmp_path, datasets={"ChEMBL": 1.0}, tokenizer_max_length=512)

    def test_prepare_data(self, dm, mock_smiles_data):
        with unittest.mock.patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = mock_smiles_data
            dm.prepare_data()

            assert len(dm.datasets) == 1
            assert isinstance(dm.dataset, ChainDataset)

    def test_prepare_data_weighted(self, dm_weighted, mock_smiles_data):
        with unittest.mock.patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = mock_smiles_data

            dm_weighted.prepare_data()
            assert len(dm_weighted.datasets) == 1
            assert isinstance(dm_weighted.dataset, MultiplexedSamplingDataset)

    def test_train_dataloader(self, dm, mock_smiles_data):
        with unittest.mock.patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = mock_smiles_data

            dm.prepare_data()
            dm.setup()

            dataloader = dm.train_dataloader()
            assert isinstance(dataloader, DataLoader)
            assert dataloader.batch_size == 8

            batch = next(iter(dataloader))
            assert isinstance(batch, BatchEncoding)
            assert isinstance(batch["input_ids"], Tensor)
