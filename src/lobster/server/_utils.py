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


def get_share_path():
    import sys
    from os.path import abspath, dirname, exists, join, split

    path = abspath(dirname(__file__))
    starting_points = [path]
    if not path.startswith(sys.prefix):
        starting_points.append(sys.prefix)
    for path in starting_points:
        # walk up, looking for prefix/share/tiled
        while path != "/":
            share_tiled = join(path, "share", "lobster")
            if exists(join(share_tiled, ".identifying_file_42688fa5f3a65e2caa822dec544d5694")):
                # We have the found the right directory,
                # or one that is trying very hard to pretend to be!
                return share_tiled
            path, _ = split(path)
    # Give up
    return ""


# Package managers can just override this with the appropriate constant
SHARE_LOBSTER_PATH = get_share_path()
