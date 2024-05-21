import pytest
from lobster.data import CynoPKClearanceLightningDataModule


@pytest.fixture(autouse=True)
def dm(tmp_path):
    datamodule = CynoPKClearanceLightningDataModule(
        root=".",
        columns=["fv_heavy", "fv_light"],
        target_columns=["cl_mean"],
        batch_size=64,
        lengths=(0.8, 0.1, 0.1),
        download=True,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    return datamodule


class TestCynoPKDatamodule:
    def test_setup(self, dm: CynoPKClearanceLightningDataModule):
        assert len(dm._train_dataset) == 66
        assert len(dm._val_dataset) == 8
        assert len(dm._test_dataset) == 8

        batch = next(iter(dm.train_dataloader()))

        assert len(batch) == 2
