import pytest
from lobster.model import LobsterPMLM, PPIClassifier


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
