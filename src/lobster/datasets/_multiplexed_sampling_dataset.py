from typing import Iterator, Literal, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class MultiplexedSamplingDataset(IterableDataset):
    """Dataset that samples from multiple iterable datasets according to specified weights."""

    def __init__(
        self,
        datasets: Sequence[IterableDataset | Dataset],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
        mode: Literal["max_size_cycle", "min"] = "min",
        max_size: int | None = None,
    ):
        """
        Initialize multiplexed dataset.

        Important: We cannot take dataset sizes into account
                   If you know the dataset sizes, you can adjust the weights accordingly.
                   See examples below.


        Examples:
            ```py

            from collections import Counter
            from lobster.datasets import MultiplexedSamplingDataset
            from torch.utils.data import Dataset, IterableDataset

            class IterableStringDataset(IterableDataset):
                def __init__(self, data):
                    self.data = data

                def __iter__(self):
                    for item in self.data:
                        yield item

            datasets= [
                IterableStringDataset(["Banana"] * 100),
                IterableStringDataset(["Apple"] * 500),
                IterableStringDataset(["Orange"] * 1000),
            ]

            # 1. Sample with equal probability from each dataset
            >>> dataset = MultiplexedSamplingDataset(datasets, seed=0, max_size=2000)
            >>> samples = list(dataset)
            >>> Counter(samples)
            Counter({'Orange': 711, 'Banana': 648, 'Apple': 641})

            # 2. Sample more proportionally from each dataset
            >>> dataset = MultiplexedSamplingDataset(datasets, weights=[100,500,1000], seed=0, max_size=2000)
            >>> samples = list(dataset)
            >>> Counter(samples), len(samples)
            Counter({'Orange': 1287, 'Apple': 610, 'Banana': 103})

            # 3. Sample with equal probability from each dataset, but stop after the shortest dataset is done
            >>> dataset = MultiplexedSamplingDataset(datasets, seed=0, mode="min")
            >>> samples = list(dataset)
            >>> Counter(samples)
            Counter({'Orange': 106, 'Banana': 100, 'Apple': 98})

            # 4. Sample with equal probability from each dataset, but cycle through the longest dataset
            >>> dataset = MultiplexedSamplingDataset(datasets, seed=0, mode="max_size_cycle")
            >>> samples = list(dataset)
            >>> Counter(samples)
            Counter({'Orange': 1000, 'Banana': 925, 'Apple': 913})
            ```

        Parameters
        ----------
        datasets : Sequence[IterableDataset | Dataset]
            Sequence of datasets to sample from.
        weights : Sequence[float] | None, optional
            Optional sampling weights for datasets. If None, all datasets are sampled equally.
        seed : int | None, optional
            Optional random seed for reproducibility.
        mode : Literal["max_size_cycle", "min"], default="max_size_cycle"
            Ignored if `max_size` is not None.
            Sampling mode.
             - "min" stops after the shortest iterable is done (default)
                Effect: downsamples the longer datasets to match the shortest dataset.
             - "max_size_cycle" stops after the longest iterable is done, while cycling through the rest.
                Effect: cycles through the shorter datasets to match the longest dataset.
            Similar to https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html
        max_size : int | None, optional
            Optional maximum number of samples to return. If None, uses `mode` to determine stopping condition.
        """
        self.datasets = datasets
        self.max_size = max_size

        if mode not in {"max_size_cycle", "min"}:
            raise ValueError("mode must be 'max_size_cycle' or 'min'")

        self.mode = mode

        if weights is not None:
            if len(weights) != len(datasets):
                raise ValueError("Number of weights must match number of datasets")
            self.weights = weights
        else:
            weights = [1.0] * len(datasets)

        self.weights = torch.tensor([w / sum(weights) for w in weights])
        self.seed = seed
        self._worker_info = None
        self._shared_seed = None
        self._generator = None

    def _get_shared_seed(self) -> int:
        """Get seed shared across workers for consistent sampling."""
        if self._shared_seed is None:
            base_seed = self.seed if self.seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()
            self._shared_seed = base_seed
        return self._shared_seed

    def _get_iterator(self, dataset: IterableDataset) -> Iterator:
        """Get iterator for a dataset with proper worker sharding."""
        seed = self._get_shared_seed()

        # Single worker
        if self._worker_info is None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)
            return iter(dataset)

        # Multiple workers
        worker_id = self._worker_info.id
        worker_seed = seed + worker_id
        self._generator = torch.Generator()
        self._generator.manual_seed(worker_seed)
        iterator = iter(dataset)

        for _ in range(worker_id):
            next(iterator)

        return iterator

    def __iter__(self):
        """Iterate over samples from datasets according to weights."""
        self._worker_info = get_worker_info()
        iterators = {dataset: self._get_iterator(dataset) for dataset in self.datasets}

        # Calculate per-worker sample limit if max_size is specified
        samples_per_worker = None
        if self.max_size is not None:
            if self._worker_info is not None:
                # Distribute max_size among workers
                num_workers = self._worker_info.num_workers
                worker_id = self._worker_info.id

                # Calculate base samples per worker and remainder
                base_samples = self.max_size // num_workers
                remainder = self.max_size % num_workers

                # Distribute remainder among workers
                # Workers with ID < remainder get one extra sample
                if worker_id < remainder:
                    samples_per_worker = base_samples + 1
                else:
                    samples_per_worker = base_samples
            else:
                # Single worker gets all samples
                samples_per_worker = self.max_size

        # Counter for samples yielded by this worker
        samples_yielded = 0

        # If max_size is set, it takes precedence over mode
        if samples_per_worker is not None:
            while samples_yielded < samples_per_worker:
                chosen_idx = torch.multinomial(self.weights, 1, generator=self._generator).item()
                chosen_dataset = self.datasets[chosen_idx]
                try:
                    yield next(iterators[chosen_dataset])
                    samples_yielded += 1
                except StopIteration:
                    # Reset this iterator and continue if we're still under max_size
                    iterators[chosen_dataset] = self._get_iterator(chosen_dataset)

        # Otherwise, use the specified mode
        elif self.mode == "min":
            while True:
                chosen_idx = torch.multinomial(self.weights, 1, generator=self._generator).item()
                chosen_dataset = self.datasets[chosen_idx]
                try:
                    yield next(iterators[chosen_dataset])
                except StopIteration:
                    break
        else:  # max_size_cycle mode
            consumed = torch.zeros(len(self.datasets), dtype=torch.bool)

            while not consumed.all():
                chosen_idx = torch.multinomial(self.weights, 1, generator=self._generator).item()
                chosen_dataset = self.datasets[chosen_idx]
                try:
                    yield next(iterators[chosen_dataset])
                except StopIteration:
                    # Mark this dataset as consumed
                    consumed[chosen_idx] = True

                    # If all datasets have been consumed at least once, we're done
                    if consumed.all():
                        break

                    # Otherwise, reset this iterator and continue
                    iterators[chosen_dataset] = self._get_iterator(chosen_dataset)
