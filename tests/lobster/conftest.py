import logging

import pytest
from lobster.data import NegLogDataModule


@pytest.fixture(autouse=True)
def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def ppi_datamodule(tmp_path_factory):
    root = tmp_path_factory.mktemp("neglog")
    return NegLogDataModule(
        root=root,
        download=False,
        lengths=[0.7, 0.2, 0.1],
        truncation_seq_length=50,
    )
