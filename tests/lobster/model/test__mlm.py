import os
import shutil
import tempfile

import onnx
import pytest
import torch
from torch import Size, Tensor

from lobster.model import LobsterPMLM


@pytest.fixture(scope="module", autouse=True)
def manage_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    yield temp_dir  # provide the fixture value

    # After test session: remove the temporary directory and all its contents
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def model():
    model = LobsterPMLM(model_name="MLM_mini", mask_percentage=0.1, initial_mask_percentage=0.8)
    model.eval()
    return model


@pytest.fixture(scope="module")
def esmc():
    model = LobsterPMLM(model_name="esmc")
    model.eval()
    return model


class TestLobsterPMLM:
    def test_sequences_to_latents(self, model):
        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 72])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device

    def test_sequences_to_latents_esmc(self, esmc):
        inputs = ["ACDAC"]
        outputs = esmc.sequences_to_latents(inputs)

        assert len(outputs) == 30

        assert outputs[0].shape == Size([1, 512, 960])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == esmc.device

    def test_onnx(self, model):
        input_ids = torch.randint(0, 2, (4, 512)).long()  # (B, L)
        attention_mask = torch.randint(0, 2, (4, 512)).long()  # (B, L)

        hidden_states = model(input_ids, attention_mask)

        assert hidden_states.shape == torch.Size([4, 4, 512, 72])
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            "model.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "hidden_states": {0: "batch", 1: "layer", 2: "sequence", 3: "features"},
            },
            # opset_version=11,
        )

        assert os.path.exists("model.onnx")

        onnx_model = onnx.load("model.onnx")
        checked = onnx.checker.check_model(onnx_model, full_check=True)
        graph = onnx.helper.printable_graph(onnx_model.graph)

        assert checked is None  # no exception raised
        assert isinstance(graph, str)

    def test_dynamic_masking(self, model):
        assert model._initial_mask_percentage is not None

    # def test_load_from_s3(self):

    #     model = LobsterPMLM.load_from_checkpoint(
    #         "s3://prescient-pcluster-data/freyn6/models/pmlm/prod/2023-10-30T15-23-25.795635/last.ckpt"
    #     )

    #     assert model.config.hidden_size == 384


def test_mlm_checkpoint(tmp_path):
    print(f"{tmp_path=}")
    model = LobsterPMLM("MLM_mini")

    for k, v in model.named_parameters():
        torch.nn.init.normal_(v)

    model.save_pretrained(tmp_path / "checkpoint")

    model2 = LobsterPMLM(str(tmp_path / "checkpoint"))

    for (k1, v1), (k2, v2) in zip(model.named_parameters(), model2.named_parameters()):
        assert k1 == k2
        assert torch.equal(v1, v2)
        assert not torch.equal(v2, torch.zeros_like(v2)), f"{k1=}, {k2=}"

    assert torch.equal(model.model.lm_head.bias, model2.model.lm_head.bias)

    input = torch.randn(2, 72)
    output = model.model.lm_head.decoder(input)
    output2 = model2.model.lm_head.decoder(input)

    diff = output - output2
    print(f"{diff.abs().max()=}")

    torch.testing.assert_close(output, output2)
