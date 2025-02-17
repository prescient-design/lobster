from typing import Iterator, Optional, Sequence

import torch
from torch.utils.data import Sampler

from lobster.constants import WEIGHTED_CONCAT_SAMPLER_CHUNK_SIZE


class WeightedConcatSampler(Sampler[int]):
    """A sampler that handles weighted sampling from concatenated datasets.
    Supports oversampling (weight > 1) and undersampling (weight < 1).
    If undersampling, samples are drawn without replacement.
    If oversampling, samples are drawn with replacement.

    Parameters
    ----------
    dataset_sizes : Sequence[int]
        The sizes of each individual dataset in the concatenated dataset
    weights : Sequence[float]
        The weights to use for sampling from each dataset.
        weight > 1: oversample with replacement
        weight = 1: shuffles the dataset
        0 < weight < 1: undersample without replacement
    generator : Optional[torch.Generator], default=None
        Generator used for random sampling. If None, a new generator will be created.

    Example
    -------
    >>> datasets = [dataset1, dataset2, dataset3]  # Individual datasets
    >>> concat_dataset = ConcatDataset(datasets)
    >>> # dataset1 will be undersampled to 50%, dataset2 sampled once,
    >>> # dataset3 will be oversampled 3x with replacement
    >>> sampler = WeightedConcatSampler(
    ...     dataset_sizes=[len(d) for d in datasets],
    ...     weights=[0.5, 1.0, 3.0],
    ...     generator=torch.Generator().manual_seed(0)
    ... )
    >>> dataloader = DataLoader(concat_dataset, batch_size=32, sampler=sampler)
    """

    def __init__(
        self, dataset_sizes: Sequence[int], weights: Sequence[float], generator: Optional[torch.Generator] = None
    ) -> None:
        if len(dataset_sizes) != len(weights):
            raise ValueError("Number of datasets and weights must match")

        if any(w <= 0 for w in weights):
            raise ValueError("All weights must be positive")

        self.dataset_sizes = dataset_sizes
        self.weights = weights
        self.num_datasets = len(dataset_sizes)

        # for mapping indices
        self.cumulative_sizes = torch.tensor([0] + list(dataset_sizes[:-1]), dtype=torch.int64).cumsum(0)

        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)

        self.generator = generator

        # number of samples for each dataset based on weights and
        # the original dataset sizes
        self.samples_per_dataset = [int(size * weight) for size, weight in zip(dataset_sizes, weights)]

    def __iter__(self) -> Iterator[int]:
        indices = []

        for dataset_idx in range(self.num_datasets):
            samples_needed = self.samples_per_dataset[dataset_idx]
            dataset_size = self.dataset_sizes[dataset_idx]
            weight = self.weights[dataset_idx]
            dataset_indices = []

            oversample = weight > 1

            for _ in range(samples_needed // WEIGHTED_CONCAT_SAMPLER_CHUNK_SIZE):
                # If oversampling, draw with replacement
                if oversample:
                    chunk = torch.randint(
                        high=dataset_size,
                        size=(WEIGHTED_CONCAT_SAMPLER_CHUNK_SIZE,),
                        generator=self.generator,
                        dtype=torch.int64,
                    )
                # If undersampling or weight=1, draw without replacement
                else:
                    remaining = min(WEIGHTED_CONCAT_SAMPLER_CHUNK_SIZE, dataset_size)
                    chunk = torch.randperm(dataset_size, generator=self.generator)[:remaining]

                dataset_indices.append(chunk)

            # Handle remainder
            remainder = samples_needed % WEIGHTED_CONCAT_SAMPLER_CHUNK_SIZE
            if remainder > 0:
                if oversample:
                    chunk = torch.randint(
                        high=dataset_size, size=(remainder,), generator=self.generator, dtype=torch.int64
                    )
                else:
                    remaining = min(remainder, dataset_size)
                    chunk = torch.randperm(dataset_size, generator=self.generator)[:remaining]

                dataset_indices.append(chunk)

            # Combine chunks and offset indices
            if dataset_indices:
                combined_indices = torch.cat(dataset_indices)
                combined_indices += self.cumulative_sizes[dataset_idx]
                indices.append(combined_indices)

        # Combine all indices and shuffle
        all_indices = torch.cat(indices)
        rand_perm = torch.randperm(len(all_indices), generator=self.generator)

        return iter(all_indices[rand_perm].tolist())

    def __len__(self) -> int:
        return sum(self.samples_per_dataset)
