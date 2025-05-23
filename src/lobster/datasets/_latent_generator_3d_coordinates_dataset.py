from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset
from lobster.transforms import Transform


class LatentGeneratorPinderIterableDataset(HuggingFaceIterableDataset):
    """
    Iterable Dataset for the LatentGenerator tokenized Pinder protein-protein structure complexes  dataset.

    This dataset provides access to protein structural tokens from LatentGenerator from the Pinder structural complexed dataset.

    Supports both streaming mode (default) and full download mode based on the 'download' parameter.
    When streaming, the dataset will load samples on-the-fly without downloading the
    entire dataset first, which is more memory-efficient for large datasets.

    Citation:
    https://www.biorxiv.org/content/10.1101/2024.07.17.603980v4

    Example:
    --------
    ```python
    # Basic usage with default parameters (streaming mode)
    dataset = LatentGeneratorPinderIterableDataset()
    sample = next(iter(dataset))
    print(sample)
    # Output: [ "ft ec ec hp ek bt bt ek . da da ek da da ec...

    # Iterate through multiple samples
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"Sample {i+1}: {sample[:50]}...")
    ```
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "val", "test"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        download: bool = False,
        shuffle: bool = False,
        split: str = "train",
        shuffle_buffer_size: int = 1000,
        columns: list[str] | None = None,
    ):
        """
        Initialize the LatentGeneratorPinderIterableDataset.

        Parameters
        ----------
        root : str or Path or None, optional
            Root directory where the dataset is stored or will be downloaded.
            If None, uses the default cache directory.
        transform : Callable or Transform or None, optional
            Optional transform to be applied on a sample, such as tokenization
            or tensor conversion.
        download : bool, optional
            If True, downloads the full dataset before streaming. If False,
            streams the dataset on-the-fly (more memory efficient). Default is False.
        shuffle : bool, optional
            If True, shuffles the dataset. Default is True.
        split : str, optional
            Which split of the dataset to use. Must be one of 'train' or 'test'.
            Default is 'train'.
        shuffle_buffer_size : int, optional
            Buffer size for shuffling the dataset. Default is 1000.
        columns : list of str or None, optional
            List of columns to load from the dataset. If None, loads all columns.
            Default is None.
        """
        super().__init__(
            dataset_name="Sidney-Lisanza/LG_tokens_pinder",
            root=root,
            transform=transform,
            keys=["lg_token_string"] if columns is None else columns,
            split=split,
            shuffle=shuffle,
            download=download,
            shuffle_buffer_size=shuffle_buffer_size,
        )

    def _passes_type_check(self, sample: tuple[str]) -> bool:
        return all(isinstance(s[0], str) for s in sample)
