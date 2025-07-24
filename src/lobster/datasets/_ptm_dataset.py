from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import pandas as pd
import pooch
from torch.utils.data import Dataset

from lobster.transforms import Transform


T = TypeVar("T")

class PTMDataset(Dataset):

    """
    PTM Dataset from PTM-mamba paper (Nature Methods, 2024).
    
    Automatically downloads PTM labels from Zenodo if not already cached.
    
    Parameters
    ----------
    root : str or Path or None, optional
        Root directory for caching. If None, uses default cache.
    download : bool, optional
        If True, download the dataset if not present (default: True).
    transform : Callable or Transform or None, optional
        Optional transform to apply to samples.
    columns : Sequence of str or None, optional
        Columns to use from CSV.
    """

    def __init__(
        self,
        root: str | Path | None = None,  
        *,
        download: bool = True,
        transform: Callable | Transform | None = None,  
        columns: Sequence[str] | None = None,
    ) -> None:
        super().__init__()

        if root is None:
            root = pooch.os_cache("lobster")
        
        if isinstance(root, str):
            root = Path(root)

        self.root = root.resolve()  

        url = "https://zenodo.org/records/14794992/files/ptm_labels.csv?download=1"
        
        if download:
            pooch.retrieve(
                url=url,
                fname="ptm_labels.csv",
                path=self.root / self.__class__.__name__,
                progressbar=True,
            )
        
        csv_path = self.root / self.__class__.__name__ / "ptm_labels.csv"
        
        self.data = pd.read_csv(csv_path)

        self.columns = ["protein_id", "position", "ptm_type", "sequence"] if columns is None else columns

        self.transform = transform

        self._x = list(self.data[self.columns].apply(tuple, axis=1))


    
    def __len__(self) ->int:
        
        return len(self._x)
    
    def __getitem__(self,index:int) ->tuple[str, ...] | str:

        x = self._x[index]

        if len(x)==1:

            x = x[0]
        
        if self.transform is not None:

            x = self.transform(x)
            
           

        return x
        
