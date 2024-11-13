import os
import shutil
import tempfile

import pytest
from lobster.model import LobsterCBMPMLM
from torch import Size, Tensor


@pytest.fixture(scope="module", autouse=True)
def manage_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    yield temp_dir  # provide the fixture value

    # After test session: remove the temporary directory and all its contents
    shutil.rmtree(temp_dir)


class TestLobsterCBMPMLM:
    def test_sequences_to_latents(self):
        model = LobsterCBMPMLM(model_name="MLM_mini")
        model.eval()

        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 72])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device

    @pytest.mark.skip("Requires s3 access")
    def test_load_from_s3(self):
        model = LobsterCBMPMLM.load_from_checkpoint(
            "s3://prescient-pcluster-data/prescient_plm/models_to_test_4/CBM_24.ckpt"
        )

        assert model.config.hidden_size == 408
