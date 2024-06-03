import pandas as pd
import torch
from lobster.data import GeminiDataFrameLightningDataModule
from lobster.model import GeminiModel


class TestGemini:
    def test_forward(self):
        df = pd.read_csv(
            ""
        )
        dm = GeminiDataFrameLightningDataModule(data=df, batch_size=4)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))

        model = GeminiModel(
            model_name="MLM_mini",
            diff_channel_0="diff",
            diff_channel_1="add",
            diff_channel_2="mul",
        )
        model.eval()
        with torch.inference_mode():
            output = model._compute_loss(batch)
        assert output is not None
        loss, preds, ys = output

        assert isinstance(loss, torch.Tensor)
        assert isinstance(preds, torch.Tensor)
        assert isinstance(ys, torch.Tensor)

        model = GeminiModel(model_name="esm2_t6_8M_UR50D")
        model.eval()
        with torch.inference_mode():
            output = model._compute_loss(batch)
        assert output is not None
        loss, preds, ys = output

        assert isinstance(loss, torch.Tensor)
        assert isinstance(preds, torch.Tensor)
        assert isinstance(ys, torch.Tensor)
