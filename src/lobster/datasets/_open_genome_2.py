from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset
from lobster.transforms import Transform


class OpenGenome2IterableDataset(HuggingFaceIterableDataset):
    """
    Iterable Dataset for the OpenGenome2 dataset (https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1)

    This dataset provides access to genomic sequences from OpenGenome2, used to pre-train Evo2,
    which includes a diverse collection of prokaryotic genomes, eukaryotic genomes,
    metagenomic data, organelle genomes, and enriched functional regions.

    The dataset contains approximately 8.84 trillion nucleotides across various
    biological domains and is available on Hugging Face as 'arcinstitute/opengenome2'.

    Dataset composition:
    - Prokaryotic genomes: 357B nucleotides from 113,379 representative genomes
    - Eukaryotic genomes: 6.98T nucleotides from 15,032 genomes
    - Metagenomic sequencing data: 854B non-redundant nucleotides
    - Organelle genomes: 2.82B nucleotides from 32,240 organelles (17,613 mitochondria,
    12,856 chloroplasts, 1,751 plastids, 18 apicoplasts, 1 cyanelle, 1 kinetoplast)
    - Eukaryotic functional regions: 602B nucleotides focused around coding genes

    The dataset offers 'train', 'validation', and 'test' splits.

    Supports both streaming mode (default) and full download mode based on the 'download' parameter.
    Strong recommendation to use streaming mode (download=False) given the giant dataset size

    Example:
    --------
    ```python
    # Basic usage in streaming mode (default)
    dataset = OpenGenome2IterableDataset()
    sample = next(iter(dataset))
    print(sample)

    # Using with data transforms
    from lobster.transforms import SequenceToTensor
    dataset = OpenGenome2IterableDataset(
        root="./data",
        transform=SequenceToTensor(),
        download=True  # Download entire dataset first (NOT RECOMMENDED)
    )

    # Iterate through multiple samples
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"Sample {i+1}: {sample[:50]}...")
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "validation", "test"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        download: bool = False,
        shuffle: bool = True,
        split: str = "train",
        shuffle_buffer_size: int = 1000,
        limit: int | None = None,
    ):
        """
        Initialize the OpenGenome2IterableDataset.

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
            Which split of the dataset to use. Must be one of 'train', 'validation', or 'test'.
            Default is 'train'.
        shuffle_buffer_size : int, optional
            Buffer size for shuffling streaming datasets. Default is 1000.
        limit : int or None, optional
            Limit the number of samples to load.
        """
        super().__init__(
            dataset_name="arcinstitute/opengenome2",
            root=root,
            transform=transform,
            keys=["text"],  # NOTE - might be other columns later in streaming
            split=split,
            shuffle=shuffle,
            download=download,
            shuffle_buffer_size=shuffle_buffer_size,
            limit=limit,
        )

    def _process_sample(self, sample: tuple[str]) -> str:
        """Process sample from dataset."""
        # Return the sequence string directly
        return sample[0]

    def _passes_type_check(self, sample: tuple[str]) -> bool:
        """Type check to filter out invalid samples."""
        # Filter out samples that aren't strings or are empty
        return all(isinstance(s, str) and len(s) > 0 for s in sample)
