from collections import Counter

import pytest
import torch
from lobster.data import WeightedConcatSampler
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


@pytest.fixture
def sampler():
    return WeightedConcatSampler(
        dataset_sizes=[10, 10, 10], weights=[0.5, 1.0, 3.0], generator=torch.Generator().manual_seed(0)
    )


@pytest.fixture
def concat_dataset():
    return ConcatDataset(
        [TensorDataset(torch.zeros(10)), TensorDataset(torch.ones(10)), TensorDataset(torch.ones(10) * 2)]
    )


class TestWeightedConcatSampler:
    def test__init__(self, sampler):
        assert sampler.num_datasets == 3
        assert torch.equal(sampler.cumulative_sizes, torch.tensor([0, 10, 20]))

    def test__len__(self, sampler):
        assert len(sampler) == 45

    def test__iter__(self, sampler):
        indices = list(sampler)

        assert len(indices) == 45

    def test_dataloader_integration(self, sampler, concat_dataset):
        dataloader = DataLoader(concat_dataset, batch_size=5, sampler=sampler)

        samples = []
        for batch in dataloader:
            samples.extend(batch[0].tolist())

        sample_count = Counter(samples)

        assert sample_count[0.0] == 5
        assert sample_count[1.0] == 10
        assert sample_count[2.0] == 30
