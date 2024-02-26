from ._clm import PrescientPCLM
from ._mlm import PrescientPMLM
from ._rlm import PrescientPRLM

model_typer = {
    "PrescientPMLM": PrescientPMLM,
    "PrescientPCLM": PrescientPCLM,
    "PrescientPRLM": PrescientPRLM,
}
