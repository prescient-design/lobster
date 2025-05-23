from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

from lobster.datasets._huggingface_iterable_dataset import HuggingFaceIterableDataset
from lobster.transforms import Transform


class AMPLIFYIterableDataset(HuggingFaceIterableDataset):
    """
    Iterable Dataset for the AMPLIFY protein language model dataset (UR100P).

    This dataset provides access to protein sequences from the AMPLIFY project
    (Fournier et al., 2024), which focuses on efficient protein language models without
    excessive scaling. The UR100P dataset contains protein sequences from various sources
    including UniProt, OAS (antibodies), and SCOP (protein structures) databases.

    The dataset is available on Hugging Face as 'chandar-lab/UR100P' and offers both
    'train' and 'test' splits.

    Supports both streaming mode (default) and full download mode based on the 'download' parameter.
    When streaming, the dataset will load samples on-the-fly without downloading the
    entire dataset first, which is more memory-efficient for large datasets.

    Citation:
    Fournier, Q., Vernon, R.M., van der Sloot, A., Schulz, B., Chandar, S., Langmead, C.J. (2024).
    Protein Language Models: Is Scaling Necessary?
    bioRxiv, doi: 10.1101/2024.09.23.614603

    Example:
    --------
    ```python
    # Basic usage with default parameters (streaming mode)
    dataset = AMPLIFYIterableDataset()
    sample = next(iter(dataset))
    print(sample)
    # Output: 'QVQLQESGPGLVKPSGTLSLTCAVSGGSISSSNWWSWVRQPPGKGLEWIGEIYHSGST...'

    # Accessing a specific data source (UniProt, OAS, or SCOP)
    oas_dataset = AMPLIFYIterableDataset(
        data_dir="OAS",  # Use antibody sequences from OAS database
        split="test",
        shuffle=False
    )

    # Using with data transforms
    from lobster.transforms import SequenceToTensor
    dataset = AMPLIFYIterableDataset(
        root="./data",
        transform=SequenceToTensor(),
        download=True  # Download entire dataset first
    )

    # Iterate through multiple samples
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        print(f"Sample {i+1}: {sample[:50]}...")
    ```
    """

    SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "test"]

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transform: Callable | Transform | None = None,
        download: bool = False,
        shuffle: bool = False,
        split: str = "train",
        data_dir: Literal["UniProt", "OAS", "SCOP"] | None = None,
        shuffle_buffer_size: int = 1000,
        limit: int | None = None,
    ):
        """
        Initialize the AMPLIFYIterableDataset.

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
        data_dir : Literal["UniProt", "OAS", "SCOP"] or None, optional
            Specific data source to use. Options include:
            - "UniProt": Reference proteomes from UniProt
            - "OAS": Antibody sequences from the OAS database
            - "SCOP": Protein structure sequences
            If None, uses all available sources.
        shuffle_buffer_size : int, optional
            Buffer size for shuffling streaming datasets. Default is 1000.
        limit : int or None, optional
            Limit the number of samples to load.
        """
        super().__init__(
            dataset_name="chandar-lab/UR100P",
            root=root,
            transform=transform,
            keys=["sequence"],
            split=split,
            shuffle=shuffle,
            data_dir=data_dir,
            download=download,
            shuffle_buffer_size=shuffle_buffer_size,
            limit=limit,
        )

    def _process_sample(self, sample: tuple[str]) -> str:
        return sample[0].replace("|", ".")

    def _passes_type_check(self, sample: tuple[str]) -> bool:
        return all(isinstance(s, str) for s in sample)
