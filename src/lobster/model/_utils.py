import torch

from ._cbmlm import LobsterCBMPMLM
from ._clm import LobsterPCLM
from ._conditioanalclassifiermlm import LobsterConditionalClassifierPMLM
from ._conditioanalmlm import LobsterConditionalPMLM
from ._mlm import LobsterPMLM
from ._pooler import LMAttentionPool1D, LMClsPooler, LMMeanPooler, LMWeightedMeanPooler

model_typer = {
    "LobsterPMLM": LobsterPMLM,
    "LobsterPCLM": LobsterPCLM,
    "LobsterConditionalPMLM": LobsterConditionalPMLM,
    "LobsterConditionalClassifierPMLM": LobsterConditionalClassifierPMLM,
    "LobsterCBMPMLM": LobsterCBMPMLM,
}

POOLERS = {
    "mean": LMMeanPooler,
    "attn": LMAttentionPool1D,
    "cls": LMClsPooler,
    "weighted_mean": LMWeightedMeanPooler,
}


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
