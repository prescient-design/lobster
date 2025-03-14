"""From Keunwoo Choi's https://code.roche.com/choik11/genie-proteinie/-/blob/k/vanilla-data/genie_proteinie/data/utils.py"""

from typing import Literal

from torch.utils.data import IterableDataset


class RoundRobinConcatIterableDataset(IterableDataset):
    """Dataset that concatenates multiple datasets. Yields one item from
    each dataset in round-robin fashion.
    """

    def __init__(self, datasets: list[IterableDataset], stopping_condition: Literal["max", "min"] = "min"):
        """Initialize concatenated dataset.

        Parameters
        ----------
        datasets : list[IterableDataset]
            List of datasets to concatenate.
        stopping_condition : Literal["max", "min"]
            If "max", the iteration will stop when the longest dataset is
            exhausted. If "min", the iteration will stop when the shortest
            dataset is exhausted.
        """
        self.datasets = datasets
        self.stopping_condition = stopping_condition

    def __iter__(self):
        """Iterate over datasets in round-robin fashion.

        Example:
        >>> dataset1 = IterableStringDataset(["Banana"] * 2)
        >>> dataset2 = IterableStringDataset(["Apple"] * 1)
        >>> dataset3 = IterableStringDataset(["Orange"] * 3)

        # If stopping_condition is "max", the iteration will stop when the
        # longest dataset is exhausted
        >>> concat_dataset = RoundRobinConcatIterableDataset(
        [dataset1, dataset2, dataset3],
        stopping_condition="max"
        )

        >>> for item in concat_dataset:
        ...     print(item)
        Banana
        Apple
        Orange
        Banana
        Orange
        Orange

        # If stopping_condition is "min", the iteration will stop when the
        # shortest dataset is exhausted
        >>> concat_dataset = RoundRobinConcatIterableDataset(
        [dataset1, dataset2, dataset3],
        stopping_condition="min"
        )

        >>> for item in concat_dataset:
        ...     print(item)
        Banana
        Apple
        Orange
        """

        # Create iterators for each dataset
        iterators = [iter(dataset) for dataset in self.datasets]

        if self.stopping_condition == "max":
            # Continue until all iterators are exhausted
            exhausted = [False] * len(iterators)
            while not all(exhausted):
                for i in range(len(iterators)):
                    if exhausted[i]:
                        continue
                    try:
                        yield next(iterators[i])
                    except StopIteration:
                        exhausted[i] = True

        elif self.stopping_condition == "min":
            # Continue until any iterator is exhausted
            while True:
                for iterator in iterators:
                    try:
                        yield next(iterator)
                    except StopIteration:
                        return
