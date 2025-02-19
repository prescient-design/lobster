from enum import Enum
from pathlib import Path
import pandas as pd
import pooch
from torch.utils.data import Dataset
from typing import Optional, Literal, Callable, Sequence, Tuple

from lobster.transforms import Transform
from lobster.constants._calm_tasks import CALM_DATA_GITHUB_URL, Species, Task, TASK_SPECIES, FUNCTION_ZENODO_BASE_URL, FILE_HASHES


class CalmPropertyDataset(Dataset):
    def __init__(
        self,
        task: Task | str,
        root: str | Path | None = None,
        *,
        species: Optional[Species | str] = None,
        split: Optional[Literal["train", "validation", "test"]] = None,
        download: bool = True,
        transform: Optional[Callable | Transform] = None,
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
        self.transform = transform
        
        if root is None:
            root = pooch.os_cache("lbster")
        if isinstance(root, str):
            root = Path(root)
        self.root = root.resolve()
        
        # Determine file name and URL based on task type
        if task in [Task.FUNCTION_BP, Task.FUNCTION_CC, Task.FUNCTION_MF]:
            function_type = task.value.split('_')[1].lower()  # Extract bp, cc, or mf
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
            raise FileNotFoundError(
                f"Data file {file_path} not found and download=False"
            )
            
        if str(file_path).endswith('.parquet'):
            self.data = pd.read_parquet(file_path)
        elif str(file_path).endswith('.tsv'):
            self.data = pd.read_csv(file_path, sep='\t')
        else:
            self.data = pd.read_csv(file_path)
        
        if columns is None:
            if task == Task.FUNCTION_BP:
                columns = ['sequence', "GO:0051092", "GO:0016573", "GO:0031146", "GO:0071427", "GO:0006613"]
            elif task == Task.FUNCTION_CC:
                columns = ['sequence', "GO:0022627", "GO:0000502", "GO:0034705", "GO:0030665", "GO:0005925"]
            elif task == Task.FUNCTION_MF:
                columns = ['sequence', "GO:0004843", "GO:0004714", "GO:0003774", "GO:0008227", "GO:0004866"]
            elif task == Task.LOCALIZATION:
                columns = ['Sequence', 'Cell membrane', 'Cytoplasm', 'Endoplasmic reticulum',
                          'Extracellular', 'Golgi apparatus', 'Lysosome/Vacuole',
                          'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid']
            elif task == Task.MELTOME:
                columns = ['sequence', 'melting_temperature']
            elif task == Task.SOLUBILITY:
                columns = ['cds', 'solubility']
            elif task == Task.PROTEIN_ABUNDANCE:
                columns = ['cds', 'abundance']
            elif task == Task.TRANSCRIPT_ABUNDANCE:
                columns = ['cds', 'logtpm']
            else:
                columns = list(self.data.columns)
        
        self.columns = columns
        self._x = self.data[self.columns].apply(tuple, axis=1)
    
    def __getitem__(self, index: int) -> Tuple:
        x = self._x[index]
        
        if len(x) == 1:
            x = x[0]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x
    
    def __len__(self) -> int:
        return len(self._x)