from dataclasses import asdict

import pandas as pd
import pytest
from omegaconf import OmegaConf

from prescient_plm.deployment.embedding import PredictConfig, predict  # isort:skip


@pytest.fixture
def dataframe():
    return pd.DataFrame({"sequence": ["MRLIPLMRLIPL", "DIVMQLM"]})


def test_embeddings(dataframe):

    config = PredictConfig(
        checkpoint_path=None,
        model_name="esm2_t6_8M_UR50D",
    )
    config = OmegaConf.create(asdict(config))

    out = predict(dataframe=dataframe, config=config)

    assert isinstance(out, pd.DataFrame)
    assert all(col in out.columns for col in dataframe.columns)
    assert len(out) == len(dataframe)
    assert len(out.columns) > len(dataframe.columns)
