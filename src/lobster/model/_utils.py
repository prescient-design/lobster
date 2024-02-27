from ._clm import PrescientPCLM
from ._mlm import PrescientPMLM

model_typer = {
    "PrescientPMLM": PrescientPMLM,
    "PrescientPCLM": PrescientPCLM,
}
