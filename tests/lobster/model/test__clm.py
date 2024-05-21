import os

import pandas as pd
import torch
from lobster.data import DataFrameDatasetInMemory
from lobster.model import LobsterPCLM
from torch import Size, Tensor
from torch.utils.data import DataLoader

CUR_DIR = os.path.dirname(__file__)


class TestLobsterPMLM:
    def test_sequences_to_latents(self):
        model = LobsterPCLM(model_name="CLM_mini")

        inputs = ["ACDAC"]
        outputs = model.sequences_to_latents(inputs)

        assert len(outputs) == 4

        assert outputs[0].shape == Size([1, 512, 32])

        assert isinstance(outputs[0], Tensor)

        assert outputs[0].device == model.device

    def test_get_batch_likelihood(self):
        model = LobsterPCLM(model_name="CLM_mini")

        inputs = ["ACDAC", "ACCAC", "ACC"]
        log_likelihood_ref = model.sequences_to_log_likelihoods(inputs)

        dataset = DataFrameDatasetInMemory(
            pd.DataFrame({"inputs": inputs}),
            transform_fn=model._transform_fn,
            columns=["inputs"],
        )

        batch_likelihood = []
        for batch in DataLoader(dataset, batch_size=64, shuffle=False):
            batch_likelihood.extend(model.batch_to_log_likelihoods(batch))

        log_likelihood_batch = torch.stack(batch_likelihood)
        assert torch.allclose(
            log_likelihood_ref, log_likelihood_batch, rtol=1e-05, atol=1e-08
        )
