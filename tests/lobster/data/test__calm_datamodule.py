import unittest.mock

import pytest
from pandas import DataFrame

from lobster.data import CalmLightningDataModule
from lobster.datasets import CalmIterableDataset


@pytest.fixture
def dm(tmp_path):
    with unittest.mock.patch("lobster.datasets._huggingface_iterable_dataset.load_dataset") as mock_load_dataset:
        mock_load_dataset.return_value = DataFrame({"sequence": 10 * ["ATG"], "description": 10 * ["dna"]})

        datamodule = CalmLightningDataModule(
            root=tmp_path,
            batch_size=8,
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule


class TestCalmLightningDataModule:
    def test_prepare_data(self, dm: CalmLightningDataModule):
        assert dm._dataset is not None
        assert isinstance(dm._dataset, CalmIterableDataset)
