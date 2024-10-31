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
