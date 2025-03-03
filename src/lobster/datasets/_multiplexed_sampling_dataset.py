"""From Keunwoo Choi's https://code.roche.com/choik11/genie-proteinie/-/blob/k/vanilla-data/genie_proteinie/data/utils.py"""

from typing import Sequence

import torch
from torch.utils.data import Dataset, IterableDataset


class MultiplexedSamplingDataset(IterableDataset):
    """Dataset that samples from multiple datasets according to specified weights."""

    def __init__(
        self,
        datasets: Sequence[Dataset | IterableDataset],
        weights: Sequence[float | int] = None,
        seed: int | None = None,
    ):
        if weights is not None:
            if len(datasets) != len(weights):
                raise ValueError("Number of datasets and weights must match")
            if not all(w >= 0 for w in weights):
                raise ValueError("Weights must be non-negative")

        self.datasets = datasets
        self.seed = seed
        self.weights = weights if weights is not None else [1.0] * len(datasets)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set random seed for sampling
        if self.seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)
        else:
            self.generator = None

    def __iter__(self):
        """Iterate over samples from datasets according to weights."""

        # Create iterators for each dataset
        # Assume each dataset handles worker sharding internally
        iterators = {dataset: iter(dataset) for dataset in self.datasets}

        # Sample from datasets according to weights
        while True:
            try:
                idx = torch.multinomial(
                    torch.tensor(self.weights), 1, replacement=True, generator=self.generator
                ).item()

                chosen_dataset = self.datasets[idx]

                yield next(iterators[chosen_dataset])

            except StopIteration:
                # If any dataset is exhausted, stop iteration
                break
