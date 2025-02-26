import unittest.mock

import pytest
from pandas import DataFrame

from lobster.data import M320MLightningDataModule
from lobster.datasets import M320MDataset


@pytest.fixture
def dm(tmp_path):
    with unittest.mock.patch("pandas.read_parquet") as mock_read_parquet:
        mock_read_parquet.return_value = DataFrame({"smiles": 10 * ["CCO"], "Description": 10 * ["description"]})

        datamodule = M320MLightningDataModule(
            root=tmp_path,
            batch_size=8,
            lengths=(0.8, 0.1, 0.1),
            download=False,
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule


class TestM320MLightningDataModule:
    def test_prepare_data(self, dm: M320MLightningDataModule):
        assert dm._dataset is not None
        assert len(dm._dataset) == 10
        assert isinstance(dm._dataset, M320MDataset)

    def test_setup(self, dm: M320MLightningDataModule):
        assert len(dm._train_dataset) == 8
        assert len(dm._val_dataset) == 1
        assert len(dm._test_dataset) == 1

    def test_train_dataloader(self, dm: M320MLightningDataModule):
        dataloader = dm.train_dataloader()
        batch = next(iter(dataloader))

        assert len(batch) == 8
        assert batch[0] == "CCO"
