"""Adapted from Keunwoo Cho's https://code.roche.com/choik11/genie-proteinie/-/blob/k/vanilla-data/genie_proteinie/data/utils.py"""

import random
from typing import Iterator, Literal, Sequence

from torch.utils.data import Dataset, IterableDataset, get_worker_info


class MultiplexedSamplingDataset(IterableDataset):
    """Dataset that samples from multiple datasets according to specified weights."""

    def __init__(
        self,
        datasets: Sequence[IterableDataset | Dataset],
        weights: Sequence[float] | None = None,
        seed: int | None = None,
        mode: Literal["max_size_cycle", "min"] = "min",
    ):
        """
        Initialize multiplexed dataset.

        Parameters
        ----------
        datasets : Sequence[IterableDataset | Dataset]
            Sequence of datasets to sample from.
        weights : Sequence[float] | None, optional
            Optional sampling weights for datasets. If None, all datasets are sampled equally.
        seed : int | None, optional
            Optional random seed for reproducibility.
        mode : Literal["max_size_cycle", "min"], default="max_size_cycle"
            Sampling mode.
             - "min" stops after the shortest iterable is done (default)
             - "max_size_cycle" stops after the longest iterable is done, while cycling through the rest.
            Similar to https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.utilities.combined_loader.html
        """
        self.datasets = datasets

        if mode not in {"max_size_cycle", "min"}:
            raise ValueError("mode must be 'max_size_cycle' or 'min'")

        self.mode = mode

        if weights is not None:
            if len(weights) != len(datasets):
                raise ValueError("Number of weights must match number of datasets")
            self.weights = weights
        else:
            weights = [1.0] * len(datasets)

        self.weights = [w / sum(weights) for w in weights]
        self.seed = seed
        self._worker_info = None
        self._shared_seed = None

    def _get_shared_seed(self) -> int:
        """Get seed shared across workers for consistent sampling."""
        if self._shared_seed is None:
            base_seed = self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
            self._shared_seed = base_seed
        return self._shared_seed

    def _get_iterator(self, dataset: IterableDataset) -> Iterator:
        """Get iterator for a dataset with proper worker sharding."""
        seed = self._get_shared_seed()

        # Single worker
        if self._worker_info is None:
            random.seed(seed)
            return iter(dataset)

        # Multiple workers
        worker_id = self._worker_info.id
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        iterator = iter(dataset)

        for _ in range(worker_id):
            next(iterator)

        return iterator

    def __iter__(self):
        """Iterate over samples from datasets according to weights."""
        self._worker_info = get_worker_info()
        iterators = {dataset: self._get_iterator(dataset) for dataset in self.datasets}

        random_seed = self._get_shared_seed()
        rng = random.Random(random_seed)

        if self.mode == "min":
            while True:
                chosen_dataset = rng.choices(self.datasets, weights=self.weights, k=1)[0]
                try:
                    yield next(iterators[chosen_dataset])
                except StopIteration:
                    break
        else:
            consumed = [False] * len(self.datasets)

            while not all(consumed):
                chosen_dataset = rng.choices(list(self.datasets), weights=self.weights, k=1)[0]
                try:
                    yield next(iterators[chosen_dataset])
                except StopIteration:
                    # Mark this dataset as consumed
                    consumed[self.datasets.index(chosen_dataset)] = True

                    # If all datasets have been consumed at least once, we're done
                    if all(consumed):
                        break

                    # Otherwise, reset this iterator and continue
                    iterators[chosen_dataset] = self._get_iterator(chosen_dataset)
