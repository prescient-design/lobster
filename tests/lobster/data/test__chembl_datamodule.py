import unittest.mock

import pytest
from beignet.datasets import ChEMBLDataset
from pandas import DataFrame

from lobster.data import ChEMBLLightningDataModule


@pytest.fixture
def dm(tmp_path):
    with unittest.mock.patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = DataFrame({"smiles": 10 * ["CCO"]})

        datamodule = ChEMBLLightningDataModule(
            root=tmp_path,
            batch_size=8,
            lengths=(0.8, 0.1, 0.1),
            download=False,
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule


class TestChEMBLLightningDataModule:
    def test_prepare_data(self, dm: ChEMBLLightningDataModule):
        assert dm._dataset is not None
        assert len(dm._dataset) == 10
        assert isinstance(dm._dataset, ChEMBLDataset)

    def test_setup(self, dm: ChEMBLLightningDataModule):
        assert len(dm._train_dataset) == 8
        assert len(dm._val_dataset) == 1
        assert len(dm._test_dataset) == 1

    def test_train_dataloader(self, dm: ChEMBLLightningDataModule):
        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))

        assert len(batch) == 8
        assert batch[0] == "CCO"
