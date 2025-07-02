import itertools

def single_position_masked_sequences(input: str, mask: str = "<mask>") -> list[str]:
    return [str.join("", [aa if j != i else mask for j, aa in enumerate(input)]) for i in range(len(input))]

# python < 3.12 compat
def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


