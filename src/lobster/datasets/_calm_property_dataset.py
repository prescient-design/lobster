from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Tuple

import pandas as pd
import pooch
import torch
from torch import Tensor
from torch.utils.data import Dataset

from lobster.constants._calm_tasks import (
    CALM_DATA_GITHUB_URL,
    CALM_TASKS,
    FILE_HASHES,
    FUNCTION_ZENODO_BASE_URL,
    TASK_SPECIES,
    Species,
    Task,
)
from lobster.transforms import Transform


class CalmPropertyDataset(Dataset):
    def __init__(
        self,
        task: Task | str,
        root: str | Path | None = None,
        *,
        species: Optional[Species | str] = None,
        split: Optional[Literal["train", "validation", "test"]] = None,
        download: bool = True,
        transform_fn: Optional[Callable | Transform] = None,
        target_transform_fn: Optional[Callable | Transform] = None,
        columns: Optional[Sequence[str]] = None,
        known_hash: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(task, str):
            task = Task(task)
        if isinstance(species, str):
            species = Species(species)

        self.task = task
        self.species = species
        self.split = split
        self.transform_fn = transform_fn
        self.target_transform_fn = target_transform_fn

        self.task_type, self.num_classes = CALM_TASKS[task.value]

        if root is None:
            root = pooch.os_cache("lbster")
        if isinstance(root, str):
            root = Path(root)
        self.root = root.resolve()

        # Get file name and URL based on task type
        if task in [Task.FUNCTION_BP, Task.FUNCTION_CC, Task.FUNCTION_MF]:
            function_type = task.value.split("_")[1].lower()  # Extract bp, cc, or mf
            fname = f"calm_GO_{function_type}_middle_normal.parquet"
            url = f"{FUNCTION_ZENODO_BASE_URL}/{fname}"
            storage_fname = fname
        else:
            if task in [Task.MELTOME, Task.SOLUBILITY, Task.LOCALIZATION]:
                fname = f"{task.value}_data.csv"
                url = f"{CALM_DATA_GITHUB_URL}/{task.value}/{fname}"
                storage_fname = f"{task.value}.csv"
            elif task in [Task.PROTEIN_ABUNDANCE, Task.TRANSCRIPT_ABUNDANCE]:
                if species is None:
                    raise ValueError(f"Must specify species for {task.value} task")
                if species not in TASK_SPECIES[task]:
                    raise ValueError(f"Species {species.value} not available for {task.value} task")
                fname = f"{species.value}.csv"
                url = f"{CALM_DATA_GITHUB_URL}/{task.value}/{fname}"
                storage_fname = f"{task.value}_{species.value}.csv"

        # Get hash for the storage filename --> to ensure file has not changed
        if known_hash is None and storage_fname in FILE_HASHES:
            known_hash = FILE_HASHES[storage_fname]

        file_path = Path(self.root / self.__class__.__name__ / storage_fname)
        if download:
            file_path = pooch.retrieve(
                url=url,
                fname=storage_fname,
                known_hash=known_hash,
                path=self.root / self.__class__.__name__,
                progressbar=True,
            )
        elif not file_path.exists():
            raise FileNotFoundError(f"Data file {file_path} not found and download=False")

        if str(file_path).endswith(".parquet"):
            self.data = pd.read_parquet(file_path)
        elif str(file_path).endswith(".tsv"):
            self.data = pd.read_csv(file_path, sep="\t")
        else:
            self.data = pd.read_csv(file_path)

        if columns is None:
            if task == Task.FUNCTION_BP:
                columns = ["sequence", "GO:0051092", "GO:0016573", "GO:0031146", "GO:0071427", "GO:0006613"]
            elif task == Task.FUNCTION_CC:
                columns = ["sequence", "GO:0022627", "GO:0000502", "GO:0034705", "GO:0030665", "GO:0005925"]
            elif task == Task.FUNCTION_MF:
                columns = ["sequence", "GO:0004843", "GO:0004714", "GO:0003774", "GO:0008227", "GO:0004866"]
            elif task == Task.LOCALIZATION:
                columns = [
                    "Sequence",
                    "Cell membrane",
                    "Cytoplasm",
                    "Endoplasmic reticulum",
                    "Extracellular",
                    "Golgi apparatus",
                    "Lysosome/Vacuole",
                    "Mitochondrion",
                    "Nucleus",
                    "Peroxisome",
                    "Plastid",
                ]
            elif task == Task.MELTOME:
                columns = ["sequence", "melting_temperature"]
            elif task == Task.SOLUBILITY:
                columns = ["cds", "solubility"]
            elif task == Task.PROTEIN_ABUNDANCE:
                columns = ["cds", "abundance"]
            elif task == Task.TRANSCRIPT_ABUNDANCE:
                columns = ["cds", "logtpm"]
            else:
                columns = list(self.data.columns)

        self.columns = columns

    def __getitem__(self, index: int) -> Tuple[str | Tensor, Tensor]:
        item = self.data.iloc[index]

        x = item[self.columns[0]]  # First column is always the input sequence/data

        if self.transform_fn is not None:
            x = self.transform_fn(x)

        y_cols = self.columns[1:]
        y_values = pd.to_numeric(item[y_cols]).values

        if self.task_type == "regression":
            y = torch.tensor(y_values, dtype=torch.float32)
        else:  # multilabel tasks (localization and function prediction)
            y = torch.tensor(y_values, dtype=torch.long)

        if self.target_transform_fn is not None:
            y = self.target_transform_fn(y)

        return x, y

    def __len__(self) -> int:
        return len(self.data)
