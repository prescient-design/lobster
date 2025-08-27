from enum import Enum


class PoolingType(str, Enum):
    """Enumeration of pooling strategies for token-level embeddings."""

    MEAN = "mean"
    MAX = "max"
    CLS = "cls"
    LAST = "last"
