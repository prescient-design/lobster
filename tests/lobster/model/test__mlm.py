import os
import shutil
import tempfile

import pytest
from lobster.data import _PRESCIENT_AVAILABLE, _PRESCIENT_PLM_AVAILABLE
from lobster.model import LobsterPMLM
from torch import Size, Tensor


@pytest.fixture(scope="module", autouse=True)
def manage_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    yield temp_dir  # provide the fixture value

    # After test session: remove the temporary directory and all its contents
    shutil.rmtree(temp_dir)


class TestLobsterPMLM:
    def test_sequences_to_latents(self):
        model = LobsterPMLM(model_name="MLM_mini")
        model.eval()

        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 72])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device

    def test_dynamic_masking(self):
        model = LobsterPMLM(
            model_name="MLM_mini", mask_percentage=0.1, initial_mask_percentage=0.8
        )

        assert model._initial_mask_percentage is not None

    def test_load_from_s3(self):
        if _PRESCIENT_AVAILABLE and _PRESCIENT_PLM_AVAILABLE:
            from prescient_plm.model import PrescientPMLM

            model = PrescientPMLM.load_from_checkpoint(
                ""
            )

            assert model.config.hidden_size == 384
