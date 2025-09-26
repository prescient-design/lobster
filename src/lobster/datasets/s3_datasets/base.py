import logging
import os
from collections.abc import Callable
from typing import Any

import litdata
from litdata import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader
from upath import UPath

from lobster.constants import Modality, Split
from lobster.tokenization import (
    AminoAcidTokenizerFast,
    NucleotideTokenizerFast,
    SMILESTokenizerFast,
    UMETokenizerTransform,
)
from lobster.transforms import TokenizerTransform

logger = logging.getLogger(__name__)


class UMEStreamingDataset(StreamingDataset):
    """
    Base class for UME streaming datasets that handles tokenization and data loading
    with litdata.

    This dataset is designed to work with different biological sequence modalities
    and provides functionality for tokenizing sequences for model input. It supports
    streaming data from various sources (e.g., S3) and handles both single and multi-sequence
    tokenization scenarios.

    Class Attributes
    ---------------
    MODALITY : Modality
        The biological sequence modality this dataset handles (e.g., AMINO_ACID, SMILES)
    SPLITS : dict[Split, str]
        Mapping of dataset splits to their filepaths (e.g., {Split.TRAIN: "s3://lobster/ume-dataset/train"})
    OPTIMIZED_SPLITS : dict[Split, str] | None
        Optional mapping of dataset splits to their optimized filepaths
    SEQUENCE_KEY : str
        The key used to access the sequence data in the dataset items
    TRAIN_SIZE : int | None
        Total size of the training split, if known
    VAL_SIZE : int | None
        Total size of the validation split, if known
    LIMIT_TRAIN_SIZE : int | None
        Optional limit on the training split size
    LIMIT_VAL_SIZE : int | None
        Optional limit on the validation split size
    """

    MODALITY: Modality
    SPLITS: dict[Split, str]  # Filepaths to the splits (e.g. {Split.TRAIN: "s3://lobster/ume-dataset/train"})
    OPTIMIZED_SPLITS: dict[Split, str] | None = None  # Filepaths to the lit-optimized splits
    SEQUENCE_KEY: str

    TRAIN_SIZE: int | None = None
    VAL_SIZE: int | None = None
    LIMIT_TRAIN_SIZE: int | None = None
    LIMIT_VAL_SIZE: int | None = None

    def __init__(
        self,
        split: Split | str,
        seed: int = 0,
        cache_dir: str | None = None,
        transform_fn: Callable | None = None,
        extra_transform_fns: dict[str, Callable] | None = None,
        tokenize: bool = True,
        use_optimized: bool = False,
        use_shared_tokenizer: bool = True,
        max_length: int | None = 8192,
    ) -> None:
        """
        Initialize the UMEStreamingDataset.

        This constructor sets up the dataset with the specified configuration, including
        tokenization settings, data transformation, and caching options. It handles both
        optimized and non-optimized dataset formats.

        Parameters
        ----------
        split : Split | str
            Dataset split to use (train, validation, etc.). Can be provided as a string
            which will be converted to a Split enum.
        seed : int, default=0
            Random seed for reproducibility of data loading and shuffling
        cache_dir : str | None, default=None
            Directory for caching dataset files. If None, no caching is performed.
        transform_fn : Callable | None, default=None
            Optional function to transform sequences before tokenization. The function
            should accept a sequence and return a transformed sequence.
            Example: removing special characters from the sequence before tokenization
        extra_transform_fns : dict[str, Callable] | None, default=None
            Optional function to transform sequences. This function will be applied
            to the sequence and the outputs are included directly in the dataset item.
            Example: computing properties of the sequence for auxiliary tasks

            extra_transform_fns = {
                "rdkit_properties": <func to compute rdkit properties>,
                "other_properties": <func to compute other properties>,
             }
        tokenize : bool, default=True
            Whether to tokenize sequences. If False, raw sequences will be returned.
        use_optimized : bool, default=False
            Whether to use optimized dataset format. Requires OPTIMIZED_SPLITS to be defined.
        use_shared_tokenizer : bool, default=True
            Whether to use tokenizers that share the same vocabulary (UMETokenizerTransform)
            or whether to use individual tokenizers for each modality.
        max_length : int | None, default=8192
            Maximum sequence length for tokenization. Required if tokenize is True.
        **kwargs : Any
            Additional arguments passed to StreamingDataset constructor

        Raises
        ------
        ValueError
            If split is not found in available splits
            If use_optimized is True but OPTIMIZED_SPLITS is not defined
            If tokenize is True but max_length is None
        """
        split = Split(split) if isinstance(split, str) else split
        subsample = self._calculate_subsample_rate(split)
        s3_uri = self._get_and_validate_uri(split, use_optimized)

        super().__init__(
            s3_uri,
            item_loader=ParquetLoader() if not use_optimized else None,
            subsample=subsample,
            drop_last=True,
            shuffle=split == Split.TRAIN,
            seed=seed,
            cache_dir=cache_dir,
            force_override_state_dict=True,
        )

        self.transform_fn = transform_fn
        self.extra_transform_fns = extra_transform_fns
        self.tokenize = tokenize
        self.max_length = max_length
        self.use_optimized = use_optimized
        self.subsample = subsample
        self.use_shared_tokenizer = use_shared_tokenizer

        if tokenize:
            self._setup_tokenizers(max_length, use_shared_tokenizer=use_shared_tokenizer)
        else:
            logger.warning(
                f"Tokenization is disabled for {self.__class__.__name__}. Please make sure this is intentional."
            )

    def _calculate_subsample_rate(self, split: Split) -> float:
        """
        Calculate the subsample rate based on split and dataset size limits.

        This method determines the appropriate subsample rate to use based on the
        dataset split and any size limits that have been set. It ensures that
        the dataset size stays within specified limits while maintaining
        proportional sampling.

        Parameters
        ----------
        split : Split
            The dataset split to calculate subsample rate for

        Returns
        -------
        float
            The subsample rate to use (between 0 and 1)
            Returns 1.0 if no size limits are set for the given split
        """
        if split == Split.TRAIN and self.TRAIN_SIZE is not None and self.LIMIT_TRAIN_SIZE is not None:
            return self.LIMIT_TRAIN_SIZE / self.TRAIN_SIZE
        elif split == Split.VALIDATION and self.VAL_SIZE is not None and self.LIMIT_VAL_SIZE is not None:
            return self.LIMIT_VAL_SIZE / self.VAL_SIZE
        return 1.0

    def _get_and_validate_uri(self, split: Split, use_optimized: bool) -> str:
        """
        Get and validate the S3 URI for the given split.

        This method retrieves the appropriate URI for the dataset split and performs
        validation checks, including verifying the existence of required index files.
        It can handle both optimized and non-optimized dataset formats.

        Parameters
        ----------
        split : Split
            The dataset split to get the URI for
        use_optimized : bool
            Whether to use optimized dataset format

        Returns
        -------
        str
            The validated S3 URI for the dataset split

        Raises
        ------
        ValueError
            If OPTIMIZED_SPLITS is not defined when use_optimized is True
            If the split is not found in available splits
            If the index file is missing for an optimized dataset
        """
        if use_optimized and self.OPTIMIZED_SPLITS is None:
            raise ValueError("Please define OPTIMIZED_SPLITS when seeting `use_optimized=True`")

        splits_map = self.OPTIMIZED_SPLITS if use_optimized else self.SPLITS
        s3_uri = splits_map.get(split)

        if s3_uri is None:
            raise ValueError(f"Split {split} not found in available split: {splits_map}")

        if not UPath(os.path.join(s3_uri, "index.json")).exists():
            if not use_optimized:
                logging.info(f"Index file not found at {s3_uri}, will create one now and reuse it later")
                litdata.index_parquet_dataset(s3_uri)
            else:
                raise ValueError(
                    f"Index file not found at {s3_uri}. Is the dataset use_optimized? Hint: run optimization or select `use_optimized=False`"
                )

        return s3_uri

    def _setup_tokenizers(self, max_length: int | None, use_shared_tokenizer: bool = True) -> None:
        """
        Set up tokenizers for different modalities.

        This method initializes the tokenizer registry with appropriate tokenizers
        for each supported modality. Each tokenizer is configured with the specified
        maximum sequence length.

        Parameters
        ----------
        max_length : int | None
            Maximum sequence length for tokenization

        Raises
        ------
        ValueError
            If max_length is None
        """
        if max_length is None:
            raise ValueError("max_length must be provided when tokenize is True")

        if use_shared_tokenizer:
            self.tokenizer_registry = {
                Modality.AMINO_ACID: UMETokenizerTransform(
                    modality=Modality.AMINO_ACID, max_length=max_length, return_modality=False
                ),
                Modality.SMILES: UMETokenizerTransform(
                    modality=Modality.SMILES, max_length=max_length, return_modality=False
                ),
                Modality.NUCLEOTIDE: UMETokenizerTransform(
                    modality=Modality.NUCLEOTIDE, max_length=max_length, return_modality=False
                ),
            }
        else:
            self.tokenizer_registry = {
                Modality.AMINO_ACID: TokenizerTransform(
                    AminoAcidTokenizerFast(),
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                ),
                Modality.SMILES: TokenizerTransform(
                    SMILESTokenizerFast(),
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                ),
                Modality.NUCLEOTIDE: TokenizerTransform(
                    NucleotideTokenizerFast(),
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                ),
            }

    def __next__(self) -> dict[str, Any]:
        item: dict = super().__next__()

        sequence: str = item.pop(self.SEQUENCE_KEY)

        if sequence is None:
            return self.__next__()

        if self.transform_fn:
            sequence: str | tuple[str | None, ...] | list[str | None] | None = self.transform_fn(sequence)

        if sequence is None or (isinstance(sequence, list | tuple) and any(seq is None for seq in sequence)):
            logger.warning(
                f"Item in {self.__class__.__name__} is None or contains None (`{sequence}`). Skipping this item."
            )
            return self.__next__()

        if self.extra_transform_fns is not None:
            for key, fn in self.extra_transform_fns.items():
                transformed = fn(sequence)

                if transformed is None:
                    logger.warning(
                        f"Extra transform function {key} returned None for input `{sequence}`. Skipping this item."
                    )
                    return self.__next__()

                item[key] = transformed

        if not self.tokenize:
            return {
                "input_ids": None,
                "attention_mask": None,
                "sequence": sequence,
                "modality": self.MODALITY.value,
                "dataset": self.__class__.__name__,
                **item,
            }

        encoded = self.tokenizer_registry[self.MODALITY](sequence)

        return {
            **encoded,
            "sequence": sequence,
            "modality": self.MODALITY.value,
            "dataset": self.__class__.__name__,
            **item,
        }
