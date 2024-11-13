import re
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import pooch
from beignet.datasets import FASTADataset
from pooch import Decompress

from lobster.transforms import Transform


class CalmDataset(FASTADataset):
    """
    Dataset from Outeiral, C., Deane, C.M.
    Codon language embeddings provide strong signals for use in protein engineering.
    Nat Mach Intell 6, 170–179 (2024). https://doi.org/10.1038/s42256-024-00791-0

    Training data is a single FASTA of sequences.
    """

    def __init__(
        self,
        url: str = "http://opig.stats.ox.ac.uk/data/downloads/training_data.tar.gz",
        root: Union[str, Path] = None,
        known_hash: str | None = None,
        *,
        index: bool = True,
        test_url: Union[str, Path] = "http://opig.stats.ox.ac.uk/data/downloads/heldout.tar.gz",
        columns: Optional[Sequence[str]] = None,
        target_columns: Optional[Sequence[str]] = None,
        train: bool = True,
        download: bool = False,
        transform_fn: Union[Callable, Transform, None] = None,
        target_transform_fn: Union[Callable, Transform, None] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        url : str
            URL to the file that needs to be downloaded. Ideally, the URL
            should end with a file name (e.g., `uniref50.fasta.gz`).
        root : Union[str, Path]
            Root directory where the dataset subdirectory exists or,
            if download is True, the directory where the dataset
            subdirectory will be created and the dataset downloaded.
        path : Union[str, Path]
            Path to the dataset.
        index : bool, optional
            If `True`, caches the sequence indexes to disk for faster
            re-initialization (default: `True`).
        test_path : Union[str, Path]
            Path to the test dataset.
        transform : Callable | Transform, optional
            A `Callable` or `Transform` that that maps a sequence to a
            transformed sequence (default: `None`).
        target_transform : Callable | Transform, optional
            A `Callable` or `Transform` that maps a target (a cluster
            identifier) to a transformed target (default: `None`).

        """
        if root is None:
            root = pooch.os_cache("lbster")

        if isinstance(root, str):
            root = Path(root)

        self._root = root.resolve()
        self._train = train
        self._download = download
        if self._train:
            self._path = url
        else:
            self._path = test_url

        if isinstance(self._root, str):
            self._root = Path(self._root).resolve()

        if self._train:
            name = "training_data.fasta"
            local_path = self._root / "calm" / name
        else:
            name = "hsapiens.fasta"
            local_path = self._root / "calm" / name

        name = self.__class__.__name__.replace("Dataset", "")

        _path = pooch.retrieve(
            url if self._train else test_url,
            known_hash,
            f"{name}.fasta.gz",
            root / name,
            processor=Decompress(),
            progressbar=True,
        )

        self._pattern = re.compile(r"^Calm.+_([A-Z0-9]+)\s.+$")

        super().__init__(
            local_path,
            index=index,
        )

        self._transform_fn = transform_fn

        self._target_transform_fn = target_transform_fn
