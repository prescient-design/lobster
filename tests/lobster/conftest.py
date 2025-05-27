import logging

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


@pytest.fixture(autouse=True)
def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)


class DummyLightningModule(L.LightningModule):
    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.embedding_dim = embedding_dim

    def embed(self, batch):
        return torch.randn(batch["input_ids"].shape[0], self.embedding_dim)


class DummyDataset(Dataset):
    def __init__(self, num_samples=32, seq_len=128):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.groups = ["group1", "group2"] * (num_samples // 2)

        if num_samples % 2 == 1:
            self.groups.append("group1")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, 100, (self.seq_len,))
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "dataset": self.groups[idx],
        }


class DummyLightningDataModule(L.LightningDataModule):
    """Dummy Lightning data module for evaluation."""

    def __init__(self, num_samples=32, seq_len=128):
        super().__init__()
        self.dataset = DummyDataset(num_samples, seq_len)
        self.num_workers = 0

    def setup(self, stage: str | None = None):
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, num_workers=self.num_workers)


@pytest.fixture
def dummy_model():
    return DummyLightningModule()


@pytest.fixture
def dummy_datamodule():
    return DummyLightningDataModule()
