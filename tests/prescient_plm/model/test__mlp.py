import torch

from prescient_plm.model import RegressionHead


class TestRegressionHead:
    def test_forward(self):
        model = RegressionHead(model_name="MLM_small")
        model.eval()

        input_dim = model.input_dim

        assert input_dim == 72
        x = torch.randn([4, input_dim])

        output = model(x)

        assert output.shape == torch.Size([4, 1])
