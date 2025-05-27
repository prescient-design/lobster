import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def sample_item(items: Sequence[T], seed: int | None = None) -> T:
    """Sample one item from a sequence with optional seeding.

    Uses a local Random instance to avoid affecting global random state.

    Parameters
    ----------
    items : Sequence[T]
        Sequence to sample from
    seed : int | None, optional
        Random seed for reproducibility

    Returns
    -------
    T
        Randomly selected item

    Raises
    ------
    ValueError
        If items is empty
    """
    if not items:
        raise ValueError("Cannot sample from empty sequence")

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    return rng.choice(items)
