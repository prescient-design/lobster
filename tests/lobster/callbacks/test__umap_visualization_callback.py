import os

import numpy as np
import pytest
import torch
from lobster.callbacks import UmapVisualizationCallback
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(self, num_samples=300, feature_dim=64, num_groups=3):
        self.data = []
        self.groups = []

        for i in range(num_groups):
            group_name = f"group_{i}"
            center = np.random.randn(feature_dim) * (i + 1)
            group_data = torch.tensor(
                np.random.randn(num_samples // num_groups, feature_dim) + center, dtype=torch.float32
            )

            self.data.append(group_data)
            self.groups.extend([group_name] * (num_samples // num_groups))

        self.data = torch.cat(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"features": self.data[idx], "dataset": self.groups[idx]}


class SimpleEmbeddingModel(nn.Module):
    def __init__(self, input_dim=64, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, embedding_dim))

    def forward(self, x):
        return self.encoder(x)

    def embed_single_batch(self, batch):
        features = batch["features"]
        return self.encoder(features)


@pytest.fixture
def model():
    return SimpleEmbeddingModel()


@pytest.fixture
def dataloader():
    dataset = SimpleDataset()
    return DataLoader(dataset, batch_size=32)


def test_umap_visualization_callback(tmp_path, model, dataloader):
    """
    Test that the UmapVisualizationCallback can generate visualizations without a trainer.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory to save visualizations
    model : nn.Module
        Test embedding model
    dataloader : DataLoader
        Test data loader
    """
    output_dir = tmp_path / "umap_test"
    os.makedirs(output_dir, exist_ok=True)

    umap_callback = UmapVisualizationCallback(
        output_dir=output_dir,
        max_samples=100,
        group_by="dataset",
        group_colors={"group_0": "blue", "group_1": "red", "group_2": "green"},
    )

    output_file = output_dir / "test_umap.png"
    result_path = umap_callback.evaluate(model, dataloader, output_file=output_file)

    assert result_path.exists()


def test_umap_without_grouping(tmp_path, model, dataloader):
    """
    Test that the UmapVisualizationCallback works without grouping.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory to save visualizations
    model : nn.Module
        Test embedding model
    dataloader : DataLoader
        Test data loader
    """
    output_dir = tmp_path / "umap_test"
    os.makedirs(output_dir, exist_ok=True)

    umap_callback = UmapVisualizationCallback(
        output_dir=output_dir,
        max_samples=100,
        group_by=None,  # Disable grouping
    )

    output_file = output_dir / "no_grouping_umap.png"
    result_path = umap_callback.evaluate(model, dataloader, output_file=output_file)

    assert result_path.exists()
