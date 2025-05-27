import os

import torch


def detect_distributed_environment() -> tuple[bool, int, int]:
    """
    Detect if running in a distributed environment and return rank and world size.

    Returns
    -------
    Tuple[bool, int, int]
        A tuple containing: (is_distributed, rank, world_size)
    """
    is_distributed = False
    rank = 0
    world_size = 1

    # Check if PyTorch distributed is initialized
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_distributed = True
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # Fallback to environment variables
    elif all(var in os.environ for var in ["RANK", "WORLD_SIZE"]):
        is_distributed = True
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    return is_distributed, rank, world_size
