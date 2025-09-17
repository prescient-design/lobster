import itertools
import logging
import random
from abc import ABCMeta
from collections.abc import Iterable, Sized

from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class SizedIterable(Sized, Iterable, metaclass=ABCMeta):
    pass


def round_robin_longest(iterables: list[SizedIterable]) -> Iterable:
    """Round robin of iterables until the longest have been exhausted.

    Example
    -------
    >>> iterables = [range(5), "ABCDE", ["cat", "dog", "rabbit"]]
    >>> iterator = round_robin_longest(iterables)
    >>> list(iterator)
    [0, 'A', 'cat', 1, 'B', 'dog', 2, 'C', 'rabbit', 3, 'D', 'cat', 4, 'E', 'dog']

    Parameters
    ----------
    iterables: list[SizedIterable]
        The iterables to roundly robin

    Returns
    -------
    Iterable
        The iterator stepping round robinly through.
        Cycles through shorter iterators.

    """
    max_len = max(len(iterable) for iterable in iterables)
    iterator_cycle = itertools.cycle(
        [
            itertools.cycle(iterable) if len(iterable) < max_len else iter(iterable)
            for iterable in iterables
            if len(iterable)
        ]
    )
    for iterator in iterator_cycle:
        try:
            yield next(iterator)
        except StopIteration:
            return


class MinorityUpsampler(Sampler[int]):
    """Upsamples shorter length lists of indices by cycling through them until
    until the longer ones are exhausted.
    """

    def __init__(self, index_list: list[SizedIterable[int]]):
        self.index_list = index_list

    def __iter__(self):
        yield from round_robin_longest(self.index_list)


class RandomizedMinorityUpsampler(MinorityUpsampler):
    """Randomized version of Upsampler."""

    def _truncated_yield(self, index_list):
        # Calculate n as the total number of entries in the list of lists
        n = sum(len(sublist) for sublist in index_list)
        logger.info(f"Max number of iterations per epoch: {n}")
        count = 0
        for value in round_robin_longest(index_list):
            if count >= n:
                break
            yield value
            count += 1

    def __iter__(self):
        index_list = [idxlist.copy() for idxlist in self.index_list]
        random.shuffle(index_list)

        for idxlist_copy in index_list:
            random.shuffle(idxlist_copy)
        yield from self._truncated_yield(index_list)

    def __len__(self):
        return sum(len(sublist) for sublist in self.index_list)
