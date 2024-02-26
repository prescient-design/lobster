import pytest

from prescient_plm.data import NegLogDataModule


@pytest.fixture(scope="session")
def ppi_datamodule():
    return NegLogDataModule(
        root="s3://prescient-pcluster-data/prescient_plm/datasets/",
        download=False,
        lengths=[0.7, 0.2, 0.1],
        truncation_seq_length=50,
    )
