import pytest
import torch
from lobster.data import CynoPKClearanceLightningDataModule
from lobster.transforms import BinarizeTransform
from torch import Size


@pytest.fixture(autouse=True)
def dm(tmp_path):
    datamodule = CynoPKClearanceLightningDataModule(
        root=".",
        batch_size=64,
        lengths=(0.8, 0.1, 0.1),
        download=True,
        target_transform_fn=BinarizeTransform(threshold=6.0),
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    return datamodule


class TestBinarizeTransform:
    def test_binarize(self, dm: CynoPKClearanceLightningDataModule):
        batch = next(iter(dm.train_dataloader()))
        (fv_heavy, fv_light), target = batch
        target = target[0]

        assert target.shape == Size([64])
        assert target.dtype == torch.int64
