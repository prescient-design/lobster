"""From Keunwoo Choi's https://code.roche.com/choik11/genie-proteinie/-/blob/k/vanilla-data/genie_proteinie/data/utils.py"""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset, IterableDataset


class MultiplexedSamplingDataset(IterableDataset):
    def __init__(
        self,
        datasets: Sequence[Dataset | IterableDataset],
        weights: Sequence[float | int] = None,
        seed: int | None = None,
    ):
        """Dataset that samples from multiple datasets according to specified weights.

        This dataset implements a weighted sampling strategy across multiple source datasets.
        For each iteration, it randomly selects a source dataset according to the provided
        weights and yields the next item from that dataset. This allows creating custom
        mixing ratios of different data sources without having to physically combine them.

        Parameters
        ----------
        datasets : Sequence[Dataset | IterableDataset]
            A sequence of datasets to sample from. These can be either map-style
            datasets (implementing __getitem__ and __len__) or iterable-style
            datasets (implementing __iter__).

        weights : Sequence[float | int], optional
            Relative sampling weights for each dataset. Can be > 1.0.
            If None, equal weights will be assigned to all datasets.
            Must have the same length as datasets.
            Weights will be normalized internally so they sum to 1.0.
            Non-positive weights are not allowed.

        seed : int or None, optional
            Random seed for reproducible sampling. If None, sampling will not be
            reproducible across runs.

        Raises
        ------
        ValueError
            If the number of weights doesn't match the number of datasets,
            or if any weight is negative.

        Notes
        -----
        - If any dataset is exhausted during iteration, the entire iteration will stop.
        - When using this dataset with multiple workers, each worker will sample
          independently with the same weights but potentially different items.
        - Setting a seed ensures reproducible sampling sequences.


        Examples
        --------
        from torch.utils.data import IterableDataset
        # Create three simple iterable datasets
        datasets = [
            IterableStringDataset(["Banana"] * 100)
            IterableStringDataset(["Apple"] * 500)
            IterableStringDataset(["Orange"] * 1000)
        ]

        # Equal weighting (default)
        equal_dataset = MultiplexedSamplingDataset(datasets, seed=42)
        samples = [next(iter(equal_dataset)) for _ in range(6)]
        # Output would be a mix of fruits with roughly equal probability
        # Note that it **doesn't** take the number of items in each dataset into account
        # ['Banana', 'Orange', 'Apple', 'Orange', 'Banana','Apple']

        # Custom weighting (99% bananas)
        banana_heavy = MultiplexedSamplingDataset(
            datasets,
            weights=[0.99, 0.005, 0.005],
            seed=42
        )
        samples = [next(iter(banana_heavy)) for _ in range(6)]
        # Output would be mostly bananas
        # ['Banana', 'Banana', 'Banana', 'Banana', 'Banana', 'Banana',]


        """
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
        """Iterate over samples from datasets according to weights.

        Yields
        ------
        Any
            Items sampled from the constituent datasets according to the specified weights.

        Notes
        -----
        The iteration stops when any of the constituent datasets is exhausted,
        even if other datasets still have items available.
        """
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
