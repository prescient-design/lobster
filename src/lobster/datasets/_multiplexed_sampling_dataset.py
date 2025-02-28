"""From Keunwoo Choi's https://code.roche.com/choik11/genie-proteinie/-/blob/k/vanilla-data/genie_proteinie/data/utils.py"""
import random

from torch.utils.data import IterableDataset


class MultiplexedSamplingDataset(IterableDataset):
    """Dataset that samples from multiple datasets according to specified weights."""

    def __init__(
        self,
        datasets: dict[IterableDataset, float],
        seed: int | None = None,
    ):
        self.datasets = list(datasets.keys())
        total_weight = sum(datasets.values())
        self.weights = [w / total_weight for w in datasets.values()]
        self.seed = seed

        # For worker sharding
        self._worker_info = None
        self._shared_seed = None

    def _get_shared_seed(self) -> int:
        """Get seed shared across workers for consistent sampling."""
        if self._shared_seed is None:
            # Use provided seed or random value
            base_seed = self.seed if self.seed is not None else random.randint(0, 2**32 - 1)
            # Add epoch to seed for different permutations per epoch
            self._shared_seed = base_seed
        return self._shared_seed

    # def _get_iterator(self, dataset: IterableDataset) -> Iterator:
    #     """Get iterator for a dataset with proper worker sharding."""
    #     if self._worker_info is None:  # Single worker
    #         return iter(dataset)

    #     # Multiple workers: each worker gets a different slice of data
    #     worker_id = self._worker_info.id

    #     # Set worker seed for reproducibility
    #     worker_seed = self._get_shared_seed() + worker_id
    #     random.seed(worker_seed)

    #     # Get iterator and advance to worker's section
    #     it = iter(dataset)
    #     for _ in range(worker_id):
    #         next(it)
    #     return it

    def __iter__(self):
        """Iterate over samples from datasets according to weights."""
        # self._worker_info = get_worker_info()

        # Create iterators for each dataset
        iterators = {dataset: iter(dataset) for dataset in self.datasets}

        # Set random seed for sampling
        random_seed = self._get_shared_seed()
        rng = random.Random(random_seed)

        # Sample from datasets according to weights
        while True:
            try:
                chosen_dataset = rng.choices(self.datasets, weights=self.weights, k=1)[0]
                yield next(iterators[chosen_dataset])
            except StopIteration:
                # If any dataset is exhausted, stop iteration
                break
