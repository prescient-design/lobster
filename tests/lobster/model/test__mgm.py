import os
import shutil
import tempfile

import onnx
import pytest
import torch
from lobster.model import LobsterMGM


@pytest.fixture(scope="module", autouse=True)
def manage_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    yield temp_dir  # provide the fixture value

    # After test session: remove the temporary directory and all its contents
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def model():
    model = LobsterMGM(
        model_name="MLM_mini", mask_percentage=0.1, initial_mask_percentage=0.8, tokenizer_dir="mgm_tokenizer"
    )
    model.eval()
    return model


class TestLobsterMGM:
    def test_model_init(self, model):
        assert model is not None

        assert model._tokenizer_dir == "mgm_tokenizer"

        # inputs = ["ACDAC"]
        # outputs = model.sequences_to_latents(inputs)

        # assert len(outputs) == 4

        # assert outputs[0].shape == Size([1, 512, 72])

        # assert isinstance(outputs[0], Tensor)

        # assert outputs[0].device == model.device

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
