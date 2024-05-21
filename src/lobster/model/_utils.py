from ._clm import LobsterPCLM
from ._mlm import LobsterPMLM
from ._pooler import LMAttentionPool1D, LMClsPooler, LMMeanPooler, LMWeightedMeanPooler

model_typer = {
    "LobsterPMLM": LobsterPMLM,
    "LobsterPCLM": LobsterPCLM,
}

POOLERS = {
    "mean": LMMeanPooler,
    "attn": LMAttentionPool1D,
    "cls": LMClsPooler,
    "weighted_mean": LMWeightedMeanPooler,
}
