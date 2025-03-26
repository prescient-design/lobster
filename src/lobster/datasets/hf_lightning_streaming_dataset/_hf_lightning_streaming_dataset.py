from typing import Any, Callable, Optional, Union

import litdata
from upath import UPath

from lobster.transforms import Transform

from ._convert_hf_dataset_to_lightning import convert_huggingface_dataset_to_lightning


class HFLightningStreamingDataset(litdata.StreamingDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        root: Union[str, UPath],
        *,
        download: bool = False,
        keys: list[str] | None = None,
        split: str = "train",
        transform: Optional[Union[Callable, Transform]] = None,
        shuffle: bool = False,
        seed: int = 0,
        subsample: float = 1.0,
    ):
        """
        Initialize the LitStreamingDataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        root : str or UPath
            Base path where the processed dataset is stored.
        keys : List[str]
            List of keys that were extracted during dataset processing.
        split : str, optional
            Which split of the dataset to use, by default "train".
        transform : Callable or Transform or None, optional
            Transform to apply to each sample at runtime.
        shuffle : bool, optional
            Whether to shuffle the dataset, by default False.
        """
        self.dataset_name = dataset_name.replace("/", "__")
        self.root = UPath(root)
        self.keys = keys
        self.split = split
        self.transform = transform
        self.download = download

        self._dirpath = self.root / self.dataset_name / "processed" / split

        if not (self._dirpath / "index.json").exists():
            if self.download:
                convert_huggingface_dataset_to_lightning(
                    dataset_path,
                    self.root / self.dataset_name / "raw" / split,
                    self._dirpath,
                    keys=keys,
                )
            else:
                raise FileNotFoundError(
                    f"Dataset {self.dataset_name} has not been downloaded yet. "
                    "Set download=True to download and convert the dataset."
                )

        super().__init__(
            str(self._dirpath),
            shuffle=shuffle,
            seed=seed,
            subsample=subsample,
        )

    def __getitem__(self, index: int) -> Any:
        """Get a sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to get.

        Returns
        -------
        Any
            The sample at the given index.
        """
        sample: dict[Any, Any] = super().__getitem__(index)

        if self.keys is not None:
            sample = tuple(sample[key] for key in self.keys)
        else:
            sample = tuple(sample.values())

        if len(sample) == 1:
            sample = sample[0]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
