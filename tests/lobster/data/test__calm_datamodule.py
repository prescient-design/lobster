import pytest
import torch
from lobster.data import CalmLightningDataModule
from lobster.model import LobsterPMLM
from torch import Size


@pytest.fixture(autouse=True)
def dm(tmp_path):
    datamodule = CalmLightningDataModule(
        root="/data/bucket/freyn6/data/",
        batch_size=64,
        lengths=(0.8, 0.1, 0.1),
        download=False,
    )
    datamodule.setup(stage="fit")

    return datamodule


class TestCalmLightningDataModule:
    def test_setup(self, dm: CalmLightningDataModule):
        assert len(dm._train_dataset) == 7_017_551
        assert len(dm._val_dataset) == 877194
        assert len(dm._test_dataset) == 877193

        batch = next(iter(dm.train_dataloader()))

        assert len(batch) == 3
        assert batch["input_ids"].shape == Size([64, 1, 512])
        assert batch["attention_mask"].shape == Size([64, 1, 512])
        assert batch["labels"].shape == Size([64, 1, 512])

    def test_mlm_calm(self, dm: CalmLightningDataModule):
        model = LobsterPMLM(model_name="MLM_mini", max_length=2048)
        model.eval()

        batch = next(iter(dm.train_dataloader()))

        with torch.inference_mode():
            loss, _ = model._compute_loss(batch)

        assert isinstance(loss, torch.Tensor)
