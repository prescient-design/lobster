from typing import Literal

MoleculeNetTask = Literal["BBBP", "BACE", "HIV", "ESOL", "FreeSolv", "Lipophilicity"]

# task -> (task_type, num_classes_or_None, filename, smiles_column, target_column)
MOLECULENET_TASKS: dict[str, tuple[str, int | None, str, str, str]] = {
    "BBBP": ("binary", 2, "BBBP.csv", "smiles", "p_np"),
    "BACE": ("binary", 2, "bace.csv", "mol", "Class"),
    "HIV": ("binary", 2, "HIV.csv", "smiles", "HIV_active"),
    "ESOL": ("regression", None, "delaney-processed.csv", "smiles", "measured log solubility in mols per litre"),
    "FreeSolv": ("regression", None, "SAMPL.csv", "smiles", "expt"),
    "Lipophilicity": ("regression", None, "Lipophilicity.csv", "smiles", "exp"),
}
MOLECULENET_TASK_NAMES = set(MOLECULENET_TASKS.keys())
