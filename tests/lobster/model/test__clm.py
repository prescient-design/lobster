import os
import shutil
import tempfile

import onnx
import pandas as pd
import pytest
import torch
from lobster.data import DataFrameDatasetInMemory
from lobster.model import LobsterPCLM
from torch import Size, Tensor
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(__file__)


@pytest.fixture(scope="module", autouse=True)
def manage_temp_dir():
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)

    yield temp_dir

    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def model():
    model = LobsterPCLM(model_name="CLM_mini")
    model.eval()
    return model


class TestLobsterPCLM:
    def test_sequences_to_latents(self, model):
        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 32])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device

    def test_onnx(self, model):
        input_ids = torch.randint(0, 2, (4, 512)).long()  # (B, L)
        attention_mask = torch.randint(0, 2, (4, 512)).long()  # (B, L)

        hidden_states = model(input_ids, attention_mask)

        assert hidden_states.shape == torch.Size([4, 4, 512, 32])
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

    def test_get_batch_likelihood(self, model):
        inputs = ["ACDAC", "ACCAC", "ACC"]
        log_likelihood_ref = model.sequences_to_log_likelihoods(inputs)

        dataset = DataFrameDatasetInMemory(
            pd.DataFrame({"inputs": inputs}), transform_fn=model._transform_fn, columns=["inputs"]
        )

        batch_likelihood = []
        for batch in DataLoader(dataset, batch_size=64, shuffle=False):
            batch_likelihood.extend(model.batch_to_log_likelihoods(batch))

        log_likelihood_batch = torch.stack(batch_likelihood)
        assert torch.allclose(log_likelihood_ref, log_likelihood_batch, rtol=1e-05, atol=1e-08)
