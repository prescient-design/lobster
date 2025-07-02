import torch


def is_distributed() -> bool:
    """Check if we're in a distributed training setup.

    Returns
    -------
    bool
        True if distributed training is initialized, False otherwise
    """
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1


def get_world_size() -> int:
    """Get the world size for distributed training.

    Returns
    -------
    int
        Number of processes (GPUs) in distributed training, 1 if not distributed
    """
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of current process.

    Returns
    -------
    int
        Rank of current process, 0 if not distributed
    """
    if is_distributed():
        return torch.distributed.get_rank()
    return 0
