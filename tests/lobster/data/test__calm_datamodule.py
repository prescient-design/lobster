import pytest
import torch
from lobster.data import CalmLightningDataModule
from lobster.model import LobsterPMLM
from torch import Size


@pytest.fixture(autouse=True)
def dm(tmp_path):
    datamodule = CalmLightningDataModule(
        root="/data/bucket/freyn6/data/",
        batch_size=8,
        lengths=(0.8, 0.1, 0.1),
        train=False,
        download=False,
    )
    datamodule.setup(stage="fit")

    return datamodule


@pytest.mark.skip(reason="Need to mock.")
class TestCalmLightningDataModule:
    def test_setup(self, dm: CalmLightningDataModule):
        assert len(dm._train_dataset) == 3481
        assert len(dm._val_dataset) == 435
        assert len(dm._test_dataset) == 435

        batch, _targets = next(iter(dm.train_dataloader()))

        assert batch["input_ids"].shape == Size([8, 1, 512])
        assert batch["attention_mask"].shape == Size([8, 1, 512])
        assert batch["labels"].shape == Size([8, 1, 512])

        model = LobsterPMLM(model_name="MLM_mini", max_length=2048)
        model.eval()

        batch, _targets = next(iter(dm.train_dataloader()))

        with torch.inference_mode():
            loss, _ = model._compute_loss(batch)

        assert isinstance(loss, torch.Tensor)
