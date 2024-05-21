import pytest
from lobster.model import LobsterPMLM, PPIClassifier
from torch import Size, Tensor


@pytest.fixture(scope="module")
def ppi_clf():
    return PPIClassifier(model_name="MLM_mini")


# Test PPI initializaation
class TestPPIClassifier:
    def test___init__(self, ppi_clf):
        # test model typer load
        assert ppi_clf._base_model_type == "LobsterPMLM"
        assert isinstance(ppi_clf.base_model, LobsterPMLM)

        # Test that the encoder is frozen
        for _, param in ppi_clf.base_model.named_parameters():
            assert param.requires_grad is False

    # Test PPI forward
    def test__forward(self, ppi_clf, ppi_datamodule):  # see tests/conftest.py
        ppi_datamodule.setup(stage="fit")
        batch = next(iter(ppi_datamodule.val_dataloader()))
        output = ppi_clf.forward(batch)

        assert isinstance(output, Tensor)
        assert output.shape == Size([1])
