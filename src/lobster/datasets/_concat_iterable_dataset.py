"""From Keunwoo Choi's https://code.roche.com/choik11/genie-proteinie/-/blob/k/vanilla-data/genie_proteinie/data/utils.py"""

from torch.utils.data import IterableDataset


class ConcatIterableDataset(IterableDataset):
    """Dataset that concatenates multiple datasets sequentially."""

    def __init__(self, datasets: list[IterableDataset]):
        """Initialize concatenated dataset.

        Args:
            datasets: List of datasets to concatenate
        """
        self.datasets = datasets

    def __iter__(self):
        """Iterate over datasets in round-robin fashion."""

        # Create iterators for each dataset
        iterators = [iter(dataset) for dataset in self.datasets]

        while iterators:  # Continue until all iterators are exhausted
            # Try getting one item from each dataset in order
            for i, iterator in enumerate(iterators):
                try:
                    yield next(iterator)
                except StopIteration:
                    # If we can't get any items from this dataset, remove it
                    del iterators[i]
                    if not iterators:  # If all datasets are exhausted
                        return
