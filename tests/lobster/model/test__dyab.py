import pandas as pd
import pytest
import torch
from lobster.data import DyAbDataFrameLightningDataModule
from lobster.model import DyAbModel


class TestDyAb:
    @pytest.mark.skip("Requires s3 access")
    def test_forward(self):
        df = pd.read_csv("s3://prescient-data-dev/sandbox/liny82/C1/262_cosmo_clean.csv")
        dm = DyAbDataFrameLightningDataModule(data=df, batch_size=4)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))

        model = DyAbModel(model_name="MLM_mini", diff_channel_0="diff", diff_channel_1="add", diff_channel_2="mul")
        model.eval()
        with torch.inference_mode():
            output = model._compute_loss(batch)
        assert output is not None
        loss, preds, ys = output

        assert isinstance(loss, torch.Tensor)
        assert isinstance(preds, torch.Tensor)
        assert isinstance(ys, torch.Tensor)

        model = DyAbModel(model_name="esm2_t6_8M_UR50D")
        model.eval()
        with torch.inference_mode():
            output = model._compute_loss(batch)
        assert output is not None
        loss, preds, ys = output

        assert isinstance(loss, torch.Tensor)
        assert isinstance(preds, torch.Tensor)
        assert isinstance(ys, torch.Tensor)
