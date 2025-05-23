from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset
from lobster.transforms import Transform


class ZINCIterableDataset(HuggingFaceIterableDataset):
    """ZINC20 Molecular Dataset - a large-scale chemical compound database.

    Reference:
        ZINC20—A Free Ultralarge-Scale Chemical Database for Ligand Discovery
        John J. Irwin, Khanh G. Tang, Jennifer Young, Chinzorig Dandarchuluun,
        Benjamin R. Wong, Munkhzul Khurelbaatar, Yurii S. Moroz, John Mayfield,
        and Roger A. Sayle Journal of Chemical Information and Modeling 2020 60
        (12), 6065-6073 DOI: 10.1021/acs.jcim.0c00675

    Link:
        https://zinc20.docking.org/

    This version of ZINC hosted on HF by haydn-jones contains approximately
    1.5 billion molecular compounds represented as SMILES strings, with corresponding
    zinc_id identifiers and SELFIES representations.

    The dataset was shuffled and split into train/validation/test splits (80%/10%/10%).

    Example:
    ```python
    dataset = ZINCIterableDataset(keys=["smiles", "selfies"])
    sample = next(iter(dataset))

    # Example output:
    # ('CCCCOc1ccc(C(=O)N2CC[C@H](N[C@H](C)COC)C2)cc1',
    #  '[C][C][C][C][O][C][=C][C][=C][Branch2][Ring1][=Branch1][C][=Branch1][C][=O][N]...')
    ```
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "validation", "test"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        keys: Sequence[str] | None = None,
        split: str = "train",
        download: bool = False,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        limit: int | None = None,
    ):
        """
        Initialize the ZINC20 dataset.

        Parameters
        ----------
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample.
        keys : Sequence of str or None, optional
            List of keys to be used from the dataset. Available keys include:
            - "smiles": SMILES string representation of the molecule
            - "zinc_id": Unique identifier for each ZINC molecule
            - "selfies": SELFIES representation of the molecule
            Defaults to ["smiles"] if not specified.
        split : str, optional
            Which split of the dataset to use. Options are "train", "validation", or "test".
        download : bool, optional
            If True, download the dataset instead of streaming.
        shuffle : bool, optional
            If True, shuffles the dataset.
        shuffle_buffer_size : int, optional
            Buffer size for shuffling streaming datasets.
        limit : int or None, optional
            Limit the number of samples to load.
        """
        super().__init__(
            dataset_name="haydn-jones/ZINC20",
            root=root,
            transform=transform,
            keys=keys or ["smiles"],
            split=split,
            download=download,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            limit=limit,
        )

    def _passes_type_check(self, sample: tuple[Any, ...]) -> bool:
        """
        Check if sample passes type check.

        For ZINC20 dataset, checks that string-based samples (SMILES, SELFIES)
        are valid strings.

        Parameters
        ----------
        sample : tuple
            Tuple of sample values

        Returns
        -------
        bool
            True if sample passes type check, False otherwise
        """
        return all(isinstance(s, str) for s in sample)
