import pytest
import torch
from lobster.model import LobsterLinearProbe
from torch.autograd import gradcheck


@pytest.fixture(scope="module")
def model():
    model = LobsterLinearProbe(model_name="MLM_mini")
    model = model.double()
    return model


class TestLinearProbe:
    def test_forward(self, model):
        model.eval()

        assert model._hidden_size == 72

        x = torch.randn([4, model._hidden_size], dtype=torch.double)

        output = model(x)

        assert output.shape == torch.Size([4, 1])

    def test_gradcheck(self, model):
        torch.manual_seed(42)
        x = torch.randn([4, model._hidden_size], requires_grad=True, dtype=torch.double)
        params = [p.detach().clone().requires_grad_(True) for p in model.parameters()]

        # copy the parameters back into the model
        with torch.inference_mode():
            for p_old, p_new in zip(model.parameters(), params):
                p_old.copy_(p_new)

        # The parameters should be updated in the function (model.forward()) not inp in gradcheck
        def Model_func(inp):
            return model(inp)

        assert gradcheck(Model_func, (x,), eps=1e-6, atol=1e-4), "Autograd gradcheck for probe failed"
