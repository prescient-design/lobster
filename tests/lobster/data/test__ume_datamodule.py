from pathlib import Path

import pytest
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from lobster.data import UmeLightningDataModule


@pytest.fixture
def dm(tmp_path):
    dm = UmeLightningDataModule(root=tmp_path, datasets=["AMPLIFY"], batch_size=8, tokenizer_max_length=512)
    return dm


class TestUmeLightningDataModule:
    def test__init__(self, dm):
        assert dm._batch_size == 8
        assert dm._tokenizer_max_length == 512
        assert isinstance(dm._root, (str, Path))

    def test_train_dataloader(self, dm):
        dm.setup()

        dataloader = dm.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 8

        batch = next(iter(dataloader))
        assert isinstance(batch, BatchEncoding)
        assert isinstance(batch["input_ids"], Tensor)
