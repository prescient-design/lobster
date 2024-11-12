import os

from torch import Size

CUR_DIR = os.path.dirname(__file__)


class TestNegLogDatamodule:
    def test_setup(self, ppi_datamodule):  # see tests/conftest.py
        assert callable(ppi_datamodule._transform_fn)

        ppi_datamodule.setup(stage="fit")

        assert len(ppi_datamodule._train_dataset) == 78058
        assert len(ppi_datamodule._val_dataset) == 22302
        assert len(ppi_datamodule._test_dataset) == 11151

    def test_batch_dims(self, ppi_datamodule):
        batch = next(iter(ppi_datamodule.train_dataloader()))

        assert batch["tokens1"].shape == Size([1, 50])
        assert batch["tokens2"].shape == Size([1, 50])

        assert batch["attention_mask1"].shape == Size([1, 50])
        assert batch["attention_mask2"].shape == Size([1, 50])
