from pathlib import Path
from typing import Callable, Literal

from lobster.transforms import Transform


class ModalityConversionDataset:
    # SUPPORTED_SPLITS: ClassVar[list[str]] = ["train", "test"]

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
