from ._cbmlm import LobsterCBMPMLM
from ._clm import LobsterPCLM
from ._conditioanalclassifiermlm import LobsterConditionalClassifierPMLM
from ._conditioanalmlm import LobsterConditionalPMLM
from ._mlm import LobsterPMLM
from ._pooler import LMAttentionPool1D, LMClsPooler, LMMeanPooler, LMWeightedMeanPooler
from ._ume import UME

model_typer = {
    "LobsterPMLM": LobsterPMLM,
    "LobsterPCLM": LobsterPCLM,
    "LobsterConditionalPMLM": LobsterConditionalPMLM,
    "LobsterConditionalClassifierPMLM": LobsterConditionalClassifierPMLM,
    "LobsterCBMPMLM": LobsterCBMPMLM,
    "UME": UME,
}

POOLERS = {
    "mean": LMMeanPooler,
    "attn": LMAttentionPool1D,
    "cls": LMClsPooler,
    "weighted_mean": LMWeightedMeanPooler,
}


def get_pooler_class(pooler_name: str):
    """Return the pooler class by name, raising a helpful error if unknown."""
    if pooler_name not in POOLERS:
        available = list(POOLERS.keys())
        raise ValueError(f"Unknown pooler '{pooler_name}'. Available: {available}")
    return POOLERS[pooler_name]


def create_pooler(pooler_name: str, config):
    """Instantiate a pooler by name with the provided config."""
    pooler_cls = get_pooler_class(pooler_name)
    return pooler_cls(config=config)
