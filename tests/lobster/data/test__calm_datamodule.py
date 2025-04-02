import unittest.mock

import pytest
from pandas import DataFrame
from torch import Size

from lobster.data import CalmLightningDataModule
from lobster.datasets import CalmDataset


@pytest.fixture
def dm(tmp_path):
    with unittest.mock.patch("lobster.datasets._calm_dataset.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = DataFrame({"sequence": 10 * ["ATG"], "description": 10 * ["dna"]})

        datamodule = CalmLightningDataModule(
            root=tmp_path, batch_size=8, lengths=(0.8, 0.1, 0.1), split_mode="random_split"
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule


class TestCalmLightningDataModule:
    def test_prepare_data(self, dm: CalmLightningDataModule):
        assert dm._dataset is not None
        assert len(dm._dataset) == 10
        assert isinstance(dm._dataset, CalmDataset)

    def test_setup(self, dm: CalmLightningDataModule):
        assert len(dm._train_dataset) == 8
        assert len(dm._val_dataset) == 1
        assert len(dm._test_dataset) == 1

        batch = next(iter(dm.train_dataloader()))

        assert batch["input_ids"].shape == Size([8, 1, 512])
