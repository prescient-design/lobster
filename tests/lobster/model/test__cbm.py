import os
import shutil
import tempfile

import pytest
import torch
from torch import Size, Tensor

from lobster.model import LobsterCBMPMLM


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


def test_cbmlm_checkpoint(tmp_path):
    print(f"{tmp_path=}")
    model = LobsterCBMPMLM("MLM_mini")

    for k, v in model.named_parameters():
        torch.nn.init.normal_(v)

    model.save_pretrained(tmp_path / "checkpoint")

    model2 = LobsterCBMPMLM(str(tmp_path / "checkpoint"))

    for (k1, v1), (k2, v2) in zip(model.named_parameters(), model2.named_parameters()):
        assert k1 == k2
        assert torch.equal(v1, v2)
        assert not torch.equal(v2, torch.zeros_like(v2)), f"{k1=}, {k2=}"

    assert torch.equal(model.model.lm_head.bias, model2.model.lm_head.bias)

    input = torch.randn(2, 56)
    output = model.model.lm_head.decoder(input)
    output2 = model2.model.lm_head.decoder(input)

    diff = output - output2
    print(f"{diff.abs().max()=}")

    torch.testing.assert_close(output, output2)
