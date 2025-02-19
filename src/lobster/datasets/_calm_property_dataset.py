from enum import Enum
from pathlib import Path
import pandas as pd
import pooch
from torch.utils.data import Dataset
from typing import Optional, Literal, Callable, Sequence, Tuple

from lobster.transforms import Transform
from lobster.constants._calm_tasks import CALM_DATA_GITHUB_URL, Species, Task, TASK_SPECIES, FUNCTION_HASHES, FUNCTION_ZENODO_BASE_URL


class CalmPropertyDataset(Dataset):
    """
    Dataset from Outeiral, C., Deane, C.M. with additional function prediction support.
    Each function type (BP, CC, MF) is treated as a separate task.
    """
    
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
            
            # Use predefined hash if none provided
            if known_hash is None:
                known_hash = FUNCTION_HASHES[task.value]
            
        else:
            if task in [Task.MELTOME, Task.SOLUBILITY, Task.LOCALIZATION]:
                fname = f"{task.value}/{task.value}_data.csv"
                url = f"{CALM_DATA_GITHUB_URL}/{fname}"
            elif task in [Task.PROTEIN_ABUNDANCE, Task.TRANSCRIPT_ABUNDANCE]:
                if species is None:
                    raise ValueError(f"Must specify species for {task.value} task")
                if species not in TASK_SPECIES[task]:
                    raise ValueError(f"Species {species.value} not available for {task.value} task")
                fname = f"{task.value}/{species.value}.csv"
                url = f"{CALM_DATA_GITHUB_URL}/{task.value}/{fname}"
            else:  # species task
                if split is None and species is None:
                    raise ValueError("Must specify either split or species for species task")
                if split is not None:
                    fname = f"{split}.fasta"
                    url = f"{CALM_DATA_GITHUB_URL}/{task.value}/{split}/{fname}"
                else:
                    fname = f"{species.value}.fasta"
                    url = f"{CALM_DATA_GITHUB_URL}/{task.value}/{fname}"

            storage_fname = f"{self.__class__.__name__}_{task.value}"
            if species:
                storage_fname += f"_{species.value}"
            if split:
                storage_fname += f"_{split}"
            storage_fname += Path(fname).suffix
        
        # Download or load the file
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
            
        # Load the data based on file type
        if str(file_path).endswith('.fasta'):
            self.data = parse_fasta(file_path)
            if species:
                self.data['species'] = species.value
        elif str(file_path).endswith('.parquet'):
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
            elif task == Task.SPECIES:
                columns = ["sequence", "description"]
                if species:
                    columns.append("species")
            elif task == Task.LOCALIZATION:
                columns = ['Sequence', 'Cell membrane', 'Cytoplasm', 'Endoplasmic reticulum',
                          'Extracellular', 'Golgi apparatus', 'Lysosome/Vacuole',
                          'Mitochondrion', 'Nucleus', 'Peroxisome', 'Plastid']
            elif task == Task.MELTOME:
                columns = ['sequence', 'melting_temperature']
            elif task == Task.SOLUBILITY:
                columns = ['cds', 'solubility']
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


def parse_fasta(path: Path) -> pd.DataFrame:
    """Parse a FASTA file into a DataFrame with sequence and description columns."""
    sequences = []
    descriptions = []
    
    with open(path) as f:
        current_description = None
        current_sequence = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_description is not None:
                    sequences.append(''.join(current_sequence))
                    descriptions.append(current_description)
                    current_sequence = []
                current_description = line[1:]
            else:
                current_sequence.append(line)
        
        if current_description is not None:
            sequences.append(''.join(current_sequence))
            descriptions.append(current_description)
    
    return pd.DataFrame({
        'sequence': sequences,
        'description': descriptions
    })
