import pytest
import torch
from lobster.model import LobsterMLP
from torch.autograd import gradcheck


@pytest.fixture(scope="module")
def model():
    model = LobsterMLP(model_name="MLM_mini", max_length=128, num_labels=1, linear_probe=True)
    model = model.double()
    return model


@pytest.fixture(scope="module")
def binary_model():
    model = LobsterMLP(model_name="MLM_mini", max_length=128, num_labels=2)
    return model


@pytest.fixture(scope="module")
def multiclass_model():
    model = LobsterMLP(model_name="MLM_mini", max_length=128, num_labels=3)
    return model


class TestLobsterMLP:
    def test_forward(self, model: LobsterMLP):
        model.eval()

        assert model._hidden_size == 72

        x = torch.randn([4, model._hidden_size], dtype=torch.double)
        y = torch.randn([4], dtype=torch.double)

        output = model(x)

        assert output.shape == torch.Size([4, 1])

        loss = model.loss(output, y)
        assert loss.item() >= 0

    def test_binary_forward(self, binary_model: LobsterMLP):
        binary_model.eval()

        assert binary_model._hidden_size == 72

        x = torch.randn([4, binary_model._hidden_size])
        y = torch.ones([4, 1], dtype=torch.float)

        output = binary_model(x)

        assert output.shape == torch.Size([4, 1])

        loss = binary_model.loss(output, y)
        assert loss.item() >= 0

    def test_multiclass_forward(self, multiclass_model: LobsterMLP):
        multiclass_model.eval()

        assert multiclass_model._hidden_size == 72

        x = torch.randn([4, multiclass_model._hidden_size])
        y = torch.tensor([0, 2, 1, 0]).long()

        output = multiclass_model(x)

        assert output.shape == torch.Size([4, 3])

        loss = multiclass_model.loss(output, y)
        assert loss.item() >= 0

    def test_gradcheck(self, model: LobsterMLP):
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

        assert gradcheck(Model_func, (x,), eps=1e-6, atol=1e-4), "Autograd gradcheck for MLP failed"
