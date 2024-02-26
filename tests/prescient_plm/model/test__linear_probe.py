import torch

from prescient_plm.model import LinearProbe


class TestLinearProbe:
    def test_forward(self):
        model = LinearProbe(model_name="MLM_small")
        model.eval()

        assert model._hidden_size == 72
        x = torch.randn([4, model._hidden_size])

        output = model(x)

        assert output.shape == torch.Size([4, 1])
