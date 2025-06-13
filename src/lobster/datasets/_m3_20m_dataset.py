from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

from beignet.transforms import Transform

from lobster.constants import Modality

from ._huggingface_iterable_dataset import HuggingFaceIterableDataset


class M320MIterableDataset(HuggingFaceIterableDataset):
    """Multi-Modal Molecular (M^3) Dataset with 20M compounds.
    M3-20M is a large-scale multi-modal molecular dataset with over 20 million molecules,
    integrating SMILES, molecular graphs, 3D structures, physicochemical properties,
    and textual descriptions. Reference: https://arxiv.org/abs/2412.06847

    This dataset version contains SMILES strings and text descriptions.

    Example:

    ```python
    dataset = M320MDataset()
    next(iter(dataset))

    (
        'CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1',
        'This molecule is a sulfoxide. It is functionally related to an albendazole...'
    )
    ```
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "validation", "test"]
    MODALITY: ClassVar[Modality] = Modality.SMILES
    SEQUENCE_KEY: ClassVar[str] = "smiles"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        keys: Sequence[str] | None = None,
        split: str = "train",
        download: bool = False,
        shuffle: bool = False,
        shuffle_buffer_size: int = 1000,
        limit: int | None = None,
    ):
        """
        Initialize the M320MDataset.

        Parameters
        ----------
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
        download : bool, optional
            If True, download the dataset if not already present.
        known_hash : str or None, optional
            Known hash of the dataset file for verification.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample.
        keys : list[str] | None, optional
            Keys to use for the dataset. If None, uses the default keys.
        limit : int or None, optional
            Limit the number of samples to load.
        """
        super().__init__(
            dataset_name="karina-zadorozhny/M320M",
            root=root,
            transform=transform,
            keys=keys or [self.SEQUENCE_KEY],
            split=split,
            download=download,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            limit=limit,
        )

    def _passes_type_check(self, sample: tuple[str]) -> bool:
        return all(isinstance(s, str) for s in sample)
