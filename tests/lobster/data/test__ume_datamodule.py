from pathlib import Path
from unittest.mock import patch

import pytest
from torch import Tensor
from torch.utils.data import DataLoader

from lobster.constants import Modality, Split
from lobster.data import UMELightningDataModule

from lobster.datasets.s3_datasets.amplify import AMPLIFY

TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def mock_dataset():
    class MockAMPLIFY(AMPLIFY):
        # Override the attribute SPLITS to point to local parquet directory
        # Parquet file contains 5 samples with columns: name, sequence
        SPLITS = {
            Split.TRAIN: TEST_DATA_DIR,
            Split.VALIDATION: TEST_DATA_DIR,
            Split.TEST: TEST_DATA_DIR,
        }

    with patch("lobster.datasets.s3_datasets.AMPLIFY", MockAMPLIFY):
        yield


@pytest.fixture
def dm(tmp_path, mock_dataset):
    return UMELightningDataModule(root=tmp_path, datasets=["AMPLIFY"], batch_size=3, max_length=150)


class TestUMELightningDataModule:
    def test__init__(self, dm):
        assert dm.batch_size == 3
        assert dm.max_length == 150
        assert isinstance(dm.root, str | Path)

    def test_train_dataloader(self, dm):
        """Test that train dataloader works with mocked datasets."""
        dm.setup()

        dataloader = dm.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 3

        batch = next(iter(dataloader))

        assert isinstance(batch["input_ids"], Tensor)
        assert batch["input_ids"].shape == (3, 1, 150)
        assert batch["attention_mask"].shape == (3, 1, 150)
        assert batch["dataset"] == ["MockAMPLIFY"] * 3
        assert batch["modality"] == [Modality.AMINO_ACID] * 3
