from pathlib import Path
from unittest.mock import patch

import pytest
from torch import Tensor
from torch.utils.data import DataLoader

import lobster.datasets.s3_datasets
from lobster.constants import Modality, Split
from lobster.data import UMELightningDataModule
from lobster.datasets.s3_datasets.amplify import AMPLIFY

TEST_DATA_DIR = Path(__file__).parents[3] / "test_data/parquet"


@pytest.fixture
def mock_dataset():
    class MockDataset(AMPLIFY):
        # Override the attribute SPLITS to point to local parquet directory
        # Parquet file contains 10 rows with columns: name, sequence
        SPLITS = {
            Split.TRAIN: str(TEST_DATA_DIR),
            Split.VALIDATION: str(TEST_DATA_DIR),
            Split.TEST: str(TEST_DATA_DIR),
        }

        # Override size constants to prevent subsampling in tests
        TRAIN_SIZE = 10
        VAL_SIZE = 10
        LIMIT_TRAIN_SIZE = None
        LIMIT_VAL_SIZE = None

    with patch.object(lobster.datasets.s3_datasets, "AMPLIFY", MockDataset):
        with patch.object(lobster.datasets.s3_datasets, "ZINC", MockDataset):
            yield


@pytest.fixture
def dm(tmp_path, mock_dataset):
    return UMELightningDataModule(
        root=str(tmp_path),
        datasets=["AMPLIFY", "ZINC"],
        batch_size=3,
        max_length=150,
        dataset_kwargs={"AMPLIFY": {"use_optimized": False}, "ZINC": {"use_optimized": False}},
    )


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
        assert batch["dataset"] == ["MockDataset"] * 3
        assert batch["modality"] == [Modality.AMINO_ACID] * 3
