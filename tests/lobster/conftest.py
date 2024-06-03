import pytest
from lobster.data import NegLogDataModule


@pytest.fixture(scope="session")
def ppi_datamodule():
    return NegLogDataModule(
        root="",
        download=False,
        lengths=[0.7, 0.2, 0.1],
        truncation_seq_length=50,
    )
